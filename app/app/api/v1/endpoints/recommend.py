from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models.movie import Movie
from app.db.models.user import User
from app.db.session import get_db
from app.ml.ncf_predictor import predict_ncf
from app.schemas.recommend import RecommendationsResponse
from app.services.recommend_service import get_recommend_service

router = APIRouter()


@router.get("/{user_id}", response_model=RecommendationsResponse)
def recommend_movies(user_id: int, recommend_service=Depends(get_recommend_service)):
    recommendations = recommend_service.get_recommendations(user_id=user_id)
    return recommendations
