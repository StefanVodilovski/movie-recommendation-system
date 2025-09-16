from fastapi import Depends, APIRouter

from app.schemas.user import UserCreate, UserRead
from app.services.user_service import get_user_service

router = APIRouter()


@router.post("/add")
async def add_user(user: UserCreate, user_service=Depends(get_user_service)) -> UserRead:
    new_user = user_service.create_user(user=user)
    return new_user
