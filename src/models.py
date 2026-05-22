from pydantic import BaseModel
from typing import List


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    strategy_used: str
    heal_attempts: int
    confidence: float


class HealthResponse(BaseModel):
    status: str