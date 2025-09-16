import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models.rating import Rating
from app.db.session import SessionLocal


def update_index(session: Session):
    session.execute(text(
        "SELECT setval(pg_get_serial_sequence('ratings', 'id'), (SELECT MAX(id) FROM ratings))"
    ))
    session.commit()


def load_ratings_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    session: Session = SessionLocal()

    try:
        ratings = []
        for _, row in df.iterrows():
            rating = Rating(
                user_id=int(row["UserID"]),
                movie_id=int(row["MovieID"]),
                rating=int(row["Rating"]),
                timestamp=int(row["Timestamp"]),
            )
            ratings.append(rating)

        session.bulk_save_objects(ratings)
        session.commit()
        print(f"Inserted {len(ratings)} ratings from {csv_path}")

    finally:
        update_index(session)
        session.close()


if __name__ == "__main__":
    load_ratings_from_csv("data/train_ratings.csv")
