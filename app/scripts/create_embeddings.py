from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.db.models.movie import Movie
from sentence_transformers import SentenceTransformer


def add_movie_embeddings(batch_size: int = 500):
    session: Session = SessionLocal()
    try:
        movies = session.query(Movie).all()
        model = SentenceTransformer("all-MiniLM-L6-v2")

        for start in range(0, len(movies), batch_size):
            end = start + batch_size
            chunk = movies[start:end]
            overviews = [m.overview or "" for m in chunk]
            embeddings = model.encode(overviews, batch_size=512, show_progress_bar=True)

            for movie, emb in zip(chunk, embeddings):
                movie.embedding = emb.tolist()

            session.commit()
            print(f"Completed embedding batch {start} -> {end}")

        print(f"Added embeddings for {len(movies)} movies successfully!")

    finally:
        session.close()


if __name__ == "__main__":
    add_movie_embeddings()
