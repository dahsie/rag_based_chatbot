
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    # score : Literal['yes', 'no']
    score : str = Field(description="Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to a given question")
