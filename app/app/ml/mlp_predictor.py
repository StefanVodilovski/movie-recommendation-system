import joblib
import numpy as np
import keras
import os

from app.db.models.movie import Movie
from app.db.models.user import User

path = os.path.dirname(__file__)

MLP_MODEL_PATH = os.path.join(path, "mlp", "experiment_7/model.keras")

zip_code_encoder = joblib.load(os.path.join(path, "encoders", "Zip-code", "experiment_3", "label_encoder.pkl"))
original_language_encoder = joblib.load(os.path.join(path, "encoders", "original_language", "experiment_3", "label_encoder.pkl"))
occupation_encoder = joblib.load(os.path.join(path, "encoders", "Occupation", "experiment_3", "label_encoder.pkl"))

mlp_model = keras.models.load_model(MLP_MODEL_PATH)

EMBEDDING_DIM = 384


def build_feature_vectors(user: User, movie: Movie) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # User vector
    gender_m = 1 if user.gender == "M" else 0
    gender_f = 1 if user.gender == "F" else 0
    user_vector = np.array([
        user.id,
        user.age,
        occupation_encoder.transform([user.occupation])[0],
        zip_code_encoder.transform([user.zip_code])[0],
        gender_m,
        gender_f
    ], dtype=np.float32).reshape(1, -1)

    # Movie vector
    lang = movie.original_language
    # TODO: check this
    if lang not in original_language_encoder.classes_:
        lang = 'en'
    lang_encoded = original_language_encoder.transform([lang])[0]

    movie_vector = np.array([
        movie.action,
        movie.adventure,
        movie.animation,
        movie.children,
        movie.comedy,
        movie.crime,
        movie.documentary,
        movie.drama,
        movie.fantasy,
        movie.film_noir,
        movie.horror,
        movie.musical,
        movie.mystery,
        movie.romance,
        movie.sci_fi,
        movie.thriller,
        movie.war,
        movie.western,
        movie.adult,
        movie.popularity or 0,
        movie.vote_average or 0,
        movie.vote_count or 0,
        lang_encoded
    ], dtype=np.float32).reshape(1, -1)

    # Overview embedding
    emb = movie.embedding
    if emb is None:
        emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    else:
        emb = np.array(emb, dtype=np.float32)
        if emb.shape[0] < EMBEDDING_DIM:
            emb = np.pad(emb, (0, EMBEDDING_DIM - emb.shape[0]))
        elif emb.shape[0] > EMBEDDING_DIM:
            emb = emb[:EMBEDDING_DIM]
    overview_vector = emb.reshape(1, -1)

    return user_vector, movie_vector, overview_vector


def predict_mlp(user: User, movie: Movie) -> float:
    u_vec, m_vec, e_vec = build_feature_vectors(user, movie)
    pred = mlp_model.predict([u_vec, m_vec, e_vec], verbose=0)[0][0]
    return float(pred)
