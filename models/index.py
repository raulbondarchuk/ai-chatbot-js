from pydantic import BaseModel, field_validator


class ChatMessage(BaseModel):
    question: str = ""

    @field_validator("question", mode="before")
    def truncate_question(cls, v):  # noqa
        if isinstance(v, str):
            return v[:128]
        return v
