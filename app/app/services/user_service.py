from fastapi import Depends

from app.db.models.user import User
from app.repository.user_repository import UserRepository, get_user_repository
from app.schemas.user import UserCreate, UserRead


class UserService:
    def __init__(self, user_repo: UserRepository) -> None:
        self.user_repo = user_repo

    def create_user(self, user: UserCreate) -> UserRead:
        db_user = User(
            gender=user.gender,
            age=user.age,
            occupation=user.occupation,
            zip_code=user.zip_code
        )
        new_user = self.user_repo.save(user=db_user)

        return new_user


def get_user_service(
        user_repo: UserRepository = Depends(get_user_repository),
):
    return UserService(user_repo)
