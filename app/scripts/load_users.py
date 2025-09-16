import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.db.session import SessionLocal


def update_index(session: Session):
    session.execute(text(
        "SELECT setval(pg_get_serial_sequence('users', 'id'), (SELECT MAX(id) FROM users))"
    ))
    session.commit()


def load_users_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    session: Session = SessionLocal()

    try:
        users = []
        for _, row in df.iterrows():
            user = User(
                id=row["Id"],
                gender=row["Gender"],
                age=int(row["Age"]),
                occupation=int(row["Occupation"]),
                zip_code=row["Zip-code"],
            )
            users.append(user)

        session.bulk_save_objects(users)
        session.commit()
        print(f"Inserted {len(users)} users from {csv_path}")

    finally:
        update_index(session)
        session.close()


if __name__ == "__main__":
    load_users_from_csv("../data/train_users.csv")
