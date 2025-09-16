from fastapi import FastAPI

from app.api.v1.endpoints import user, movie, recommend, rating
from app.db.session import init_db

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    init_db()


app.include_router(user.router, prefix="/api/v1/user", tags=["User"])
app.include_router(movie.router, prefix="/api/v1/movie", tags=["Movie"])
app.include_router(recommend.router, prefix="/api/v1/recommend", tags=["Recommend"])
app.include_router(rating.router, prefix="/api/v1/rating", tags=["Rating"])
