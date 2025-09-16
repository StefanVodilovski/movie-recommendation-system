from typing import List

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.models.rating import Rating
from app.db.session import get_db


class RatingRepository:

    def __init__(self, db: Session):
        self.db = db

    def count_by_user_id(self, user_id: int) -> int:
        return self.db.query(Rating).filter(Rating.user_id == user_id).count()

    def get_rating(self, user_id: int, movie_id: int) -> Rating | None:
        return self.db.query(Rating).filter(
            Rating.user_id == user_id,
            Rating.movie_id == movie_id
        ).first()

    def create_rating(self, user_id: int, movie_id: int, rating: int, timestamp: int) -> Rating:
        new_rating = Rating(
            user_id=user_id,
            movie_id=movie_id,
            rating=rating,
            timestamp=timestamp
        )
        self.db.add(new_rating)
        self.db.commit()
        self.db.refresh(new_rating)
        return new_rating

    def update_rating(self, existing_rating: Rating, rating: int) -> Rating:
        existing_rating.rating = rating
        self.db.commit()
        self.db.refresh(existing_rating)
        return existing_rating


def get_rating_repository(db: Session = Depends(get_db)):
    return RatingRepository(db)
