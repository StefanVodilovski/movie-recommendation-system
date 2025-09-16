from pydantic import BaseModel, conint


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: conint(ge=1, le=5)
