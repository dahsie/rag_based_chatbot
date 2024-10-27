from pydantic import BaseModel, Field


class QuestionRewriter(BaseModel):
    """output parser of rewritten question for better web search"""
    question: str = Field(description='rewrite the question without any explaination')
