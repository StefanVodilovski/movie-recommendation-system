from fastapi import Depends
from sentence_transformers import SentenceTransformer

from app.db.models.movie import Movie
from app.repository.movie_repository import MovieRepository, get_movie_repository
from app.schemas.movie import MovieCreate, MovieRead


class MovieService:
    def __init__(self, movie_repo: MovieRepository) -> None:
        self.movie_repo = movie_repo
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def create_movie(self, movie: MovieCreate) -> MovieRead:
        embedding = None
        if movie.overview:
            embedding = self.embedding_model.encode(movie.overview, show_progress_bar=False).tolist()

        db_movie = Movie(
            name=movie.name,
            action=movie.action,
            adventure=movie.adventure,
            animation=movie.animation,
            children=movie.children,
            comedy=movie.comedy,
            crime=movie.crime,
            documentary=movie.documentary,
            drama=movie.drama,
            fantasy=movie.fantasy,
            film_noir=movie.film_noir,
            horror=movie.horror,
            musical=movie.musical,
            mystery=movie.mystery,
            romance=movie.romance,
            sci_fi=movie.sci_fi,
            thriller=movie.thriller,
            war=movie.war,
            western=movie.western,
            adult=movie.adult,
            original_language=movie.original_language,
            overview=movie.overview,
            popularity=movie.popularity,
            vote_average=movie.vote_average,
            vote_count=movie.vote_count,
            embedding=embedding
        )
        new_movie = self.movie_repo.save(movie=db_movie)

        return new_movie


def get_movie_service(
        movie_repo: MovieRepository = Depends(get_movie_repository),
):
    return MovieService(movie_repo)
