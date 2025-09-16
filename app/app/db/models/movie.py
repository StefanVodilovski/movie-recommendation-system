from sqlalchemy import Column, Integer, String, Boolean, Float, Text
from sqlalchemy.dialects.postgresql import ARRAY

from app.db.base import Base


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    action = Column(Boolean, nullable=False, default=False)
    adventure = Column(Boolean, nullable=False, default=False)
    animation = Column(Boolean, nullable=False, default=False)
    children = Column(Boolean, nullable=False, default=False)
    comedy = Column(Boolean, nullable=False, default=False)
    crime = Column(Boolean, nullable=False, default=False)
    documentary = Column(Boolean, nullable=False, default=False)
    drama = Column(Boolean, nullable=False, default=False)
    fantasy = Column(Boolean, nullable=False, default=False)
    film_noir = Column(Boolean, nullable=False, default=False)
    horror = Column(Boolean, nullable=False, default=False)
    musical = Column(Boolean, nullable=False, default=False)
    mystery = Column(Boolean, nullable=False, default=False)
    romance = Column(Boolean, nullable=False, default=False)
    sci_fi = Column(Boolean, nullable=False, default=False)
    thriller = Column(Boolean, nullable=False, default=False)
    war = Column(Boolean, nullable=False, default=False)
    western = Column(Boolean, nullable=False, default=False)
    adult = Column(Boolean, nullable=False, default=False)
    original_language = Column(String(10), nullable=False)
    overview = Column(Text)
    popularity = Column(Float)
    vote_average = Column(Float)
    vote_count = Column(Float)
    embedding = Column(ARRAY(Float))
