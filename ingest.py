# Langchain dependencies
import hashlib

from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing text splitter from Langchain
from langchain.text_splitter import MarkdownTextSplitter  # Importing text splitter from Langchain
from langchain_community.embeddings import OpenAIEmbeddings  # Importing OpenAI embeddings from Langchain
from langchain.schema import Document  # Importing Document schema from Langchain
from langchain_community.vectorstores import Chroma  # Importing Chroma vector store from Langchain
from langchain_ollama import OllamaEmbeddings
# from dotenv import load_dotenv  # Importing dotenv to get API key from .env file
# from langchain.chat_models import ChatOpenAI  # Import OpenAI LLM
import os  # Importing os module for operating system functionalities
import shutil  # Importing shutil module for high-level file operations


# Path to the directory to save Chroma database
CHROMA_PATH = "./db_metadata_v5"
DATA_PATH = "./docs"
global_unique_hashes = set()


def walk_through_files(path, file_extension='.txt'):
    for (dir_path, dir_names, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_documents():
    """
    Load documents from the specified directory
    Returns:
    List of Document objects:
    """
    documents = []
    for f_name in walk_through_files(DATA_PATH):
        document_loader = TextLoader(f_name, encoding="utf-8")
        documents.extend(document_loader.load())

    return documents


def hash_text(text):
    # Generate a hash value for the text using SHA-256
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()


def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.
    Args:
    documents (list[Document]): List of Document objects containing text content to split.
    Returns:
    list[Document]: List of Document objects representing the split text chunks.
    """
    # Initialize text splitter with specified parameters
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )
    """
    text_splitter = MarkdownTextSplitter(
        chunk_size=500,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
    )

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Deduplication mechanism
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash_text(chunk.page_content)
        if chunk_hash not in global_unique_hashes:
            unique_chunks.append(chunk)
            global_unique_hashes.add(chunk_hash)

    # Print example of page content and metadata for a chunk
    print(f"Unique chunks equals {len(unique_chunks)}.")
    # print(unique_chunks[:-5])

    return unique_chunks  # Return the list of split text chunks


def save_to_chroma(chunks: list[Document]):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
    chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """
    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
    """
    Function to generate vector database in chroma from documents.
    """
    documents = load_documents()  # Load documents from a source
    chunks = split_text(documents)  # Split documents into manageable chunks
    save_to_chroma(chunks)  # Save the processed data to a data store


if __name__ == "__main__":
    generate_data_store()
