from typing import Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
    question: str
    email: Optional[str]


class HubSpotForm(BaseModel):
    email: str
    firstName: Optional[str]
    lastName: Optional[str]
    phone: Optional[str]
