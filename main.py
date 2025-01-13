from fastapi.middleware.cors import CORSMiddleware

from models.index import ChatMessage
from providers.ollama import query_rag
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/public", StaticFiles(directory="public"), name="public")


@app.get("/")
async def read_root():
    # result = model.invoke(input="Hello World")
    return {"Hello": "world"}


@app.post("/chat/{chat_id}")
async def ask(chat_id: str, message: ChatMessage):
    return {"response": query_rag(message, chat_id)}
