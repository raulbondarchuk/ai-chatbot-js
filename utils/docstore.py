import json
from typing import List

from langchain_community.docstore.base import Docstore
from langchain.docstore.document import Document
import sqlite3
import uuid


class SQLiteDocStore(Docstore):
    def __init__(self, db_path="docs.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT, metadata TEXT, parsed BOOLEAN DEFAULT 0)"
        )

    def add(self, doc: Document) -> str:
        doc_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO docs VALUES (?, ?, ?, ?)",
            (doc_id, doc.page_content, json.dumps(doc.metadata), 0),
        )
        self.conn.commit()
        return doc_id

    def update_parsed_status(self, doc_ids: List[str]):
        placeholders = ','.join('?' for _ in doc_ids)
        self.conn.execute(f"UPDATE docs SET parsed=1 WHERE id IN ({placeholders})", tuple(doc_ids))
        self.conn.commit()

    def search(self, doc_id: str) -> Document:
        cur = self.conn.execute("SELECT text, metadata FROM docs WHERE id=?", (doc_id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"Doc {doc_id} not found")
        text, metadata = row
        return Document(page_content=text, metadata=json.loads(metadata))

    def list(self) -> List[Document]:
        """
        List all unparsed documents
        :return: List of Documents
        """
        cur = self.conn.execute("SELECT text, metadata, id FROM docs WHERE parsed=0")
        docs = []
        for text, meta, id in cur.fetchall():
            metadata = json.loads(meta)
            metadata['id'] = id
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def truncate(self) -> None:
        self.conn.execute("DELETE FROM docs")
        self.conn.commit()

    def delete(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM docs WHERE id=?", (doc_id,))
        self.conn.commit()
