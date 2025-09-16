from fastapi import Depends, APIRouter

from app.schemas.movie import MovieCreate, MovieRead
from app.services.movie_service import get_movie_service

router = APIRouter()


@router.post("/add")
async def add_movie(movie: MovieCreate, movie_service=Depends(get_movie_service)) -> MovieRead:
    new_movie = movie_service.create_movie(movie=movie)
    return new_movie
