from typing import Optional

from pydantic import BaseModel


class MovieRead(BaseModel):
    id: int
    name: str
    action: bool
    adventure: bool
    animation: bool
    children: bool
    comedy: bool
    crime: bool
    documentary: bool
    drama: bool
    fantasy: bool
    film_noir: bool
    horror: bool
    musical: bool
    mystery: bool
    romance: bool
    sci_fi: bool
    thriller: bool
    war: bool
    western: bool
    adult: bool
    original_language: str
    overview: str | None
    popularity: float | None
    vote_average: float | None
    vote_count: float | None

    class Config:
        from_attributes = True


class MovieCreate(BaseModel):
    name: str
    action: bool = False
    adventure: bool = False
    animation: bool = False
    children: bool = False
    comedy: bool = False
    crime: bool = False
    documentary: bool = False
    drama: bool = False
    fantasy: bool = False
    film_noir: bool = False
    horror: bool = False
    musical: bool = False
    mystery: bool = False
    romance: bool = False
    sci_fi: bool = False
    thriller: bool = False
    war: bool = False
    western: bool = False
    adult: bool = False
    original_language: str
    overview: Optional[str] = None
    popularity: Optional[float] = 0
    vote_average: Optional[float] = 0
    vote_count: Optional[float] = 0
