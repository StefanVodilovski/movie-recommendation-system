from fastapi import Depends
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models.movie import Movie
from app.db.session import get_db


class MovieRepository:

    def __init__(self, db: Session):
        self.db = db

    def save(self, movie: Movie) -> Movie:
        try:
            self.db.add(movie)
            self.db.commit()
            self.db.refresh(movie)
            return movie

        except SQLAlchemyError as e:
            self.db.rollback()
            raise

    def get_all_movies(self):
        return self.db.query(Movie).all()

    def get_movie_by_id(self, movie_id: int) -> Movie:
        return self.db.query(Movie).filter(Movie.id == movie_id).first()


def get_movie_repository(db: Session = Depends(get_db)):
    return MovieRepository(db)
