from app.db.session import init_db
from scripts.load_users import load_users_from_csv
from scripts.load_movies import load_movies_from_csv
from scripts.load_ratings import load_ratings_from_csv
from scripts.create_embeddings import add_movie_embeddings

if __name__ == "__main__":
    print("Initializing database...")
    init_db()

    print("Loading users...")
    load_users_from_csv("../data/train_users.csv")

    print("Loading movies...")
    load_movies_from_csv("../data/train_movies.csv")

    print("Loading ratings...")
    load_ratings_from_csv("../data/train_ratings.csv")

    print("Generating movie embeddings...")
    add_movie_embeddings()

    print("All data loaded successfully!")
