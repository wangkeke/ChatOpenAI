from pydantic import BaseModel
from fastapi import Query

class Message(BaseModel):
    role: str = Query(default=..., regex="^(system)|(user)|(assistant)$")
    content: str = Query(default=..., min_length=1, max_length=8000)
