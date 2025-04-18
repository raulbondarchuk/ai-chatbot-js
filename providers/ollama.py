from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages.utils import count_tokens_approximately

from models.index import ChatMessage


CHROMA_PATH = "./db_metadata_v5"

# Initialize OpenAI chat model
model = OllamaLLM(model="llama3.2:latest", temperature=0.1)


# YOU MUST - Use same embedding function as before
embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

# Prepare the database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
chat_history = {}  # approach with AiMessage/HumanMessage

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                [INST]You are a sales manager with the name 'AI Assistant'. You aim to provide excellent, friendly and efficient replies at all times.
                You will provide me with answers from the given info.
                If the answer is not included, say exactly “Hmm, I am not sure. Let me check and get back to you.”
                Refuse to answer any question not about the info.
                Never break character.
                No funny stuff.
                If a question is not clear, ask clarifying questions.
                Make sure to end your replies with a positive note.
                Do not be pushy.
                Answer should be in MD format.
                If someone asks for the price, cost, quote or similar, then reply “In order to provide you with a customized and reasonable quote, I would need a 15 minute call.
                Ready for an online meeting?[/INST]
                [INST]Answer the question based only on the following context:
                {context}[/INST]
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

# cached_chain = prompt_template | model
document_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)


async def query_rag(message: ChatMessage, session_id: str = ""):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    :param message: ChatMessage The text to query the RAG system with.
    :param session_id: str Session identifier
    :return str
    """

    if session_id not in chat_history:
        chat_history[session_id] = []

    messages = trim_messages(chat_history[session_id], strategy="last", token_counter=count_tokens_approximately,
                             max_tokens=2056, start_on="human", allow_partial=False)

    # Generate response text based on the prompt
    response_text = await document_chain.ainvoke({"context": db.similarity_search(message.question, k=3),
                                                  "question": message.question,
                                                  "chat_history": messages})

    chat_history[session_id].append(HumanMessage(content=message.question))
    chat_history[session_id].append(AIMessage(content=response_text))

    return response_text
