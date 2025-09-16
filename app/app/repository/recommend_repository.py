from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.models.movie import Movie
from app.db.models.user_recommendation import UserRecommendation
from app.db.session import get_db


class RecommendRepository:

    def __init__(self, db: Session):
        self.db = db

    def save(self, user_id: int, recommendations):
        self.delete_by_user_id(user_id=user_id)

        objects = [
            UserRecommendation(
                user_id=user_id,
                movie_id=r["movie_id"],
                score=r["score"]
            ) for r in recommendations
        ]
        self.db.add_all(objects)
        self.db.commit()

    def delete_by_user_id(self, user_id: int):
        self.db.query(UserRecommendation).filter_by(user_id=user_id).delete()

    def get_recommendations_by_user_id(self, user_id: int):
        recs = (
            self.db.query(UserRecommendation, Movie.name)
            .join(Movie, Movie.id == UserRecommendation.movie_id)
            .filter(UserRecommendation.user_id == user_id)
            .all()
        )
        return recs


def get_recommend_repository(db: Session = Depends(get_db)):
    return RecommendRepository(db)
