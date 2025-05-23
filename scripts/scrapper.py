import pathlib

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer

from utils.docstore import SQLiteDocStore

# Path to the directory to save a Chroma database
root = pathlib.Path(__file__).parent.parent.resolve()
FILE_TO_PARSE = f"{root}/data/links.txt"
DB_PATH = f"{root}/data/docs.sqlite"


def getLinks2Parse() -> list:
    try:
        with open(FILE_TO_PARSE, "r") as f:
            return [link.strip() for link in f.readlines()]
    except:
        return []


def asyncLoader(links):
    db_conn = SQLiteDocStore(db_path=DB_PATH)
    db_conn.truncate()

    loader = AsyncHtmlLoader(links)
    docs = loader.load()

    # Transform
    bs_transformer = BeautifulSoupTransformer()

    for doc in docs:
        doc.page_content = bs_transformer.remove_unwanted_classnames(doc.page_content,
                                                                     ["new-footer", "main-header",
                                                                      "main-top-block", "callback__form",
                                                                      "new-footer-bottom", "blog-article-share", "blog-article-slider",
                                                                      "blog-article-menu", "blog__subscribe",
                                                                      "main-top-block__info", "breadcrumbs"])

    html2text = Html2TextTransformer(ignore_links=True, ignore_images=True)
    docs_transformed = html2text.transform_documents(docs)

    for idx, doc in enumerate(docs_transformed):
        print("ADDED NEW DOCUMENT", db_conn.add(doc))


if __name__ == "__main__":
    ls = getLinks2Parse()
    asyncLoader(ls)
