import time

from fastapi import Depends, HTTPException

from app.repository.movie_repository import MovieRepository, get_movie_repository
from app.repository.rating_repository import get_rating_repository, RatingRepository
from app.repository.user_repository import UserRepository, get_user_repository
from app.schemas.rating import RatingRequest


class RatingService:
    def __init__(self, rating_repo: RatingRepository, user_repo: UserRepository, movie_repo: MovieRepository) -> None:
        self.rating_repo = rating_repo
        self.user_repo = user_repo
        self.movie_repo = movie_repo

    def rate_movie(self, request: RatingRequest) -> dict:

        user = self.user_repo.get_user_by_id(user_id=request.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        movie = self.movie_repo.get_movie_by_id(movie_id=request.movie_id)
        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found")

        existing_rating = self.rating_repo.get_rating(user.id, movie.id)

        if existing_rating:
            updated = self.rating_repo.update_rating(existing_rating, request.rating)
            return {"message": "Rating updated", "rating": updated.rating}
        else:
            new_rating = self.rating_repo.create_rating(
                user.id, movie.id, request.rating, int(time.time())
            )
            return {"message": "Rating created", "rating": new_rating.rating}


def get_rating_service(
        rating_repo: RatingRepository = Depends(get_rating_repository),
        user_repo: UserRepository = Depends(get_user_repository),
        movie_repo: MovieRepository = Depends(get_movie_repository),
):
    return RatingService(rating_repo, user_repo, movie_repo)
