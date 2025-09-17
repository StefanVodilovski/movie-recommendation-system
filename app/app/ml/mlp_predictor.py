import os

import keras
import numpy as np
import pandas as pd
import joblib

from app.db.models.user import User

path = os.path.dirname(__file__)

MLP_MODEL_PATH = os.path.join(path, "mlp", "experiment_7/model.keras")

mlp_model = keras.models.load_model(MLP_MODEL_PATH)

zip_code_encoder = joblib.load(os.path.join(path, "../ml/encoders", "Zip-code", "experiment_3", "label_encoder.pkl"))
original_language_encoder = joblib.load(
    os.path.join(path, "../ml/encoders", "original_language", "experiment_3", "label_encoder.pkl"))
occupation_encoder = joblib.load(
    os.path.join(path, "../ml/encoders", "Occupation", "experiment_3", "label_encoder.pkl"))

EMBEDDING_DIM = 384


def get_movies_df(engine) -> pd.DataFrame:
    query = """
        SELECT
            id AS movie_id,
            name,
            original_language,
            action, adventure, animation, children, comedy,
            crime, documentary, drama, fantasy, film_noir,
            horror, musical, mystery, romance, sci_fi,
            thriller, war, western, adult,
            COALESCE(popularity, 0) AS popularity,
            COALESCE(vote_average, 0) AS vote_average,
            COALESCE(vote_count, 0) AS vote_count,
            embedding
        FROM movies
    """
    return pd.read_sql(query, con=engine)


def add_user_features(df: pd.DataFrame, user: User) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = user.id
    df["age"] = user.age
    df["occupation"] = occupation_encoder.transform([user.occupation])[0]
    df["zip_code"] = zip_code_encoder.transform([user.zip_code])[0]
    df["gender"] = user.gender
    df = pd.get_dummies(df, columns=["gender"], dtype=np.float32)

    for col in ["gender_F", "gender_M"]:
        if col not in df.columns:
            df[col] = 0

    return df


def encode_languages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[~df["original_language"].isin(original_language_encoder.classes_), "original_language"] = "en"
    df["original_language"] = original_language_encoder.transform(df["original_language"])
    return df


def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    emb_matrix = df["embedding"].apply(
        lambda emb: np.zeros(EMBEDDING_DIM, dtype=np.float32) if emb is None
        else np.pad(np.array(emb, dtype=np.float32)[:EMBEDDING_DIM],
                    (0, max(0, EMBEDDING_DIM - len(emb))))
    )
    return np.vstack(emb_matrix.values)


def prepare_mlp_inputs(engine, user: User):
    movies_df = get_movies_df(engine)
    movies_df = add_user_features(movies_df, user)
    movies_df = encode_languages(movies_df)

    emb_matrix = build_embeddings(movies_df)

    user_matrix = movies_df[["user_id", "age", "occupation", "zip_code", "gender_F", "gender_M"]].to_numpy(np.float32)
    movie_matrix = movies_df[[
        "action", "adventure", "animation", "children", "comedy", "crime", "documentary", "drama",
        "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", "thriller",
        "war", "western", "adult", "popularity", "vote_average", "vote_count", "original_language"
    ]].to_numpy(np.float32)

    return movies_df, user_matrix, movie_matrix, emb_matrix


def predict_mlp(model_inputs):
    u_mat, m_mat, e_mat = model_inputs

    preds = mlp_model.predict([u_mat, m_mat, e_mat], verbose=0)

    return preds.flatten()
