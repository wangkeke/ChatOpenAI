from pydantic import BaseModel
from typing import Union
from fastapi import Query

class UserMessage(BaseModel):
    role: str = Query(default=..., regex="^(system)|(user)|(assistant)$")
    content: str = Query(default=..., min_length=1, max_length=8000)
