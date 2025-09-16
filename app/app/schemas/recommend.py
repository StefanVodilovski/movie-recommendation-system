from pydantic import BaseModel
from typing import List


class RecommendationSchema(BaseModel):
    movie_id: int
    name: str
    score: float


class RecommendationsResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationSchema]
