from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain


PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'


CHROMA_PATH = "./db_metadata_v5"
PROMPT_TEMPLATE = """
[INST]
I want you to act as a sales manager. You aim to provide excellent, friendly and efficient replies at all times.
Your name is “AI Assistant”. You will provide me with answers from the given info.
If the answer is not included, say exactly “Hmm, I am not sure. Let me check and get back to you.”
Refuse to answer any question not about the info. Never break character.
If a question is not clear, ask clarifying questions. Make sure to end your replies with a positive note.
Do not be pushy. If someone asks for the price, cost, quote or similar, then reply “In order to provide you with a customized and reasonable quote, I would need a 15 minute call.
Ready for an online meeting?
Answer the question based only on the following context:
{context}
Question: {question}
[/INST]
"""

PROMPT_TEMPLATE2 = """
As the Sales Manager of Geomotiv, your role is to engage potential and existing customers with in-depth knowledge of our products and services.
Your goal is to identify customer needs, present tailored solutions, and build lasting relationships to drive sales.
Use a professional, confident, and personable tone. Proactively address customer queries, showcase our unique selling points, and suggest upsell or cross-sell opportunities.
Leverage any retrieved data about specific products, services, or customer history to create a personalized and compelling sales experience.
Be attentive, persuasive, and focused on helping the customer make informed decisions.

Answer the question based only on the following context:
{context}
Question: {question}
"""


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


def rewrite_query(user_question: str, history: list):
    """
    Method to use to make USER question more relevant
    :param user_question: str
    :param history:
    :return:
    """
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-2:]])
    prompt = """Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_question}]
    
    Rewritten query: 
    """

    # prompt_template = ChatPromptTemplate.from_template(prompt)
    return model.invoke(prompt_template.format(context=context, question=user_question))


def query_rag(query_text: str, session_id: str = "") -> str:
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    :param query_text: str The text to query the RAG system with.
    :param session_id: str Session identifier
    :return str
    """

    if session_id not in chat_history:
        chat_history[session_id] = []

    # Generate response text based on the prompt
    response_text = document_chain.invoke({"context": db.similarity_search(query_text, k=3),
                                           "question": query_text,
                                           "chat_history": chat_history[session_id]})

    chat_history[session_id].append(HumanMessage(content=query_text))
    chat_history[session_id].append(AIMessage(content=response_text))

    return response_text
