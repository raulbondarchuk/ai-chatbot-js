"""
Title: Markdown Document Structurer for RAG Systems

Description:
This script processes an input Markdown (.md or .txt) document and transforms it into a more structured, retrieval-optimized format for use in a Retrieval-Augmented Generation (RAG) system.

Key Features:
- Preserves all original facts without modification or addition
- Enhances structure using headings, bullet points, and tables
- Makes implicit context explicit (e.g., decision makers, processes, eligibility criteria)
- Adds clarity to vague or ambiguous statements
- Improves readability and chunkability for embedding-based retrieval systems

Use Case:
Intended for preparing internal policy or procedural documents (e.g., HR, Education Compensation, Company Guidelines) to be ingested into RAG pipelines for chatbots or semantic search.

Note:
This transformation does not summarize or rewrite content—only restructures and clarifies existing information.

Author: [Andrei Dryzgalovich]
"""
import asyncio
import os

from dotenv import load_dotenv
from utils.index import load_documents

from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

DATA_PATH = "./tmp"


SYSTEM_PROMPT = """
You are a document structuring assistant. Your task is to transform the following Markdown document into a more structured and retrieval-friendly format for use in a Retrieval-Augmented Generation (RAG) system.
Do NOT add new facts — your goal is to make implicit rules explicit and clarify the structure without introducing new information.

Instructions:
1. Use clear hierarchical formatting:
    1.1 Headings (#, ##, ###) for organization.
    1.2 Lists, bullet points, or tables for clarity.

2. Where applicable, make implicit context explicit, such as:
    2.1 Who makes decisions
    2.2 What kind of events/courses are eligible
    2.3 How decisions are usually made

3. Include examples of email subjects and request content if they are already implied.
4. Break up long sections for better chunking.
5. Include any obvious decision criteria, timelines, or expectations if they can be inferred.
6. If motivational lines exist (e.g., "take initiative"), clarify what that might look like with examples.
7. Add Query/Answer section to the end of the file which will include up to 10 query/answer pair.

Here is the original Markdown document:

{markdown_document}

Return the transformed document in Markdown format, with no commentary.
"""

# Initialize OpenAI chat model
model = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", together_api_key=os.getenv("AI_TOGETHER_API_KEY"), temperature=0.1)
# model = ChatTogether(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", together_api_key=os.getenv("AI_TOGETHER_API_KEY"), temperature=0.1)


async def document_transform():
    """
    Load and transform documents
    :return:
    """
    docs = load_documents(DATA_PATH)

    template = ChatPromptTemplate.from_template(template=SYSTEM_PROMPT)

    for doc in docs:
        folder, file_path = os.path.split(doc.metadata["source"])
        updated_file_path = f"./docs/{file_path}.md".lower()

        prompt = template.format_messages(markdown_document=doc.page_content)

        try:
            rewritten_content = await model.ainvoke(prompt)

            with open(updated_file_path, "w") as file:
                file.write(rewritten_content.content)

            print(f"File {updated_file_path} added")
        except BaseException as ex:
            print(f"Cannot create a file {updated_file_path}: {str(ex)}")
        finally:
            # make a delay
            await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(document_transform())
