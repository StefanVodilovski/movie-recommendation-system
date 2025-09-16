import joblib
import keras
import os
import tensorflow as tf
import numpy as np

from app.db.models.movie import Movie
from app.db.models.user import User
from app.utils.extend_encoder import extend_encoder

path = os.path.dirname(__file__)

NCF_MODEL_PATH = os.path.join(path, "ncf", "experiment_21/model.keras")

encoders = joblib.load(os.path.join(path, "ncf", "experiment_21", "encoders", "user_and_movies_encoders.pkl"))
user_encoder = encoders["user_encoder"]
item_encoder = encoders["item_encoder"]

MF_DIM = 128


def mf_slice(x):
    x = tf.squeeze(x, [1])
    return x[:, :MF_DIM]


def mlp_slice(x):
    x = tf.squeeze(x, [1])
    return x[:, :MF_DIM]


@keras.saving.register_keras_serializable()
def mlp_slice_layer(x):
    return mlp_slice(x)


@keras.saving.register_keras_serializable()
def mf_slice_layer(x):
    return mf_slice(x)


ncf_model = keras.models.load_model(
    NCF_MODEL_PATH,
    custom_objects={
        "mlp_slice": mlp_slice_layer,
        "mf_slice": mf_slice_layer,
    },
    compile=False
)


def predict_ncf(user: User, movie: Movie) -> float:
    user_key = f"{user.gender}_{user.age}_{user.occupation}_{user.zip_code}"

    if user_key not in user_encoder.classes_:
        extend_encoder(user_encoder, [user_key])

    if movie.id not in item_encoder.classes_:
        extend_encoder(item_encoder, [movie.id])

    user_idx = user_encoder.transform([user_key])[0]
    movie_idx = item_encoder.transform([movie.id])[0]

    score = ncf_model.predict([np.array([user_idx]), np.array([movie_idx])], verbose=0)[0][0]
    return float(score)
