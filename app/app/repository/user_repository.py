from fastapi import Depends
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.db.session import get_db


class UserRepository:

    def __init__(self, db: Session):
        self.db = db

    def save(self, user: User) -> User:
        try:
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user

        except SQLAlchemyError as e:
            self.db.rollback()
            raise

    def get_user_by_id(self, user_id: int) -> User:
        return self.db.query(User).filter(User.id == user_id).first()


def get_user_repository(db: Session = Depends(get_db)):
    return UserRepository(db)
