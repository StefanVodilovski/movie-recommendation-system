import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models.movie import Movie
from app.db.session import SessionLocal


def update_index(session: Session):
    session.execute(text(
        "SELECT setval(pg_get_serial_sequence('movies', 'id'), (SELECT MAX(id) FROM movies))"
    ))
    session.commit()


def load_movies_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    session: Session = SessionLocal()

    try:
        movies = []
        for _, row in df.iterrows():
            movie = Movie(
                id=row["Id"],
                name=row["Name"],
                action=bool(row["Action"]),
                adventure=bool(row["Adventure"]),
                animation=bool(row["Animation"]),
                children=bool(row["Children's"]),
                comedy=bool(row["Comedy"]),
                crime=bool(row["Crime"]),
                documentary=bool(row["Documentary"]),
                drama=bool(row["Drama"]),
                fantasy=bool(row["Fantasy"]),
                film_noir=bool(row["Film-Noir"]),
                horror=bool(row["Horror"]),
                musical=bool(row["Musical"]),
                mystery=bool(row["Mystery"]),
                romance=bool(row["Romance"]),
                sci_fi=bool(row["Sci-Fi"]),
                thriller=bool(row["Thriller"]),
                war=bool(row["War"]),
                western=bool(row["Western"]),
                adult=bool(row["adult"]),
                original_language=row["original_language"],
                overview=row.get("overview"),
                popularity=row.get("popularity"),
                vote_average=row.get("vote_average"),
                vote_count=row.get("vote_count"),
            )
            movies.append(movie)

        session.bulk_save_objects(movies)
        session.commit()
        print(f"Inserted {len(movies)} movies from {csv_path}")

    finally:
        update_index(session)
        session.close()


if __name__ == "__main__":
    load_movies_from_csv("../data/train_movies.csv")
