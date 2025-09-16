from fastapi import APIRouter, Depends

from app.schemas.rating import RatingRequest
from app.services.rating_service import get_rating_service

router = APIRouter()


@router.post("/rate_movie/{user_id}/{movie_id}")
def rate_movie(
    rating_req: RatingRequest,
    rating_service=Depends(get_rating_service)
):
    result = rating_service.rate_movie(rating_req)
    return result
