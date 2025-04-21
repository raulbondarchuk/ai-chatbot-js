import os

from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document
from typing import List


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {d.metadata.get('source')}:\n\n" + d.page_content for d in docs]
        )
    )


def walk_through_files(path: str, file_extension='.txt'):
    """
    Recursively yields file paths from the given directory that match the specified extension.

    Walks through all subdirectories of the given `path` and yields the full path
    of each file that ends with the specified `file_extension`.

    Parameters:
        path (str): Root directory to start searching from.
        file_extension (str, optional): File extension to filter by. Defaults to '.txt'.

    Yields: str: Full path to each matching file.
    """
    for (dir_path, dir_names, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_documents(data_path: str, extension='.txt') -> List[Document]:
    """
    Loads documents from files in the specified directory.

    Recursively traverses all files in the `data_path` directory,
    loads them using `TextLoader`, and returns a list of `Document` objects.

    Parameters:
        data_path (str): Path to the directory containing the files.
        extension (str, optional): File extension to load.
                                   Defaults to '.txt'.

    Returns: List[Document]: A list of loaded documents.
    """
    documents = []
    for f_name in walk_through_files(data_path, extension):
        document_loader = TextLoader(f_name, encoding="utf-8")
        documents.extend(document_loader.load())

    return documents


PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
