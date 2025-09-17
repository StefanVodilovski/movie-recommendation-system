import os

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from app.db.models.user import User

path = os.path.dirname(__file__)

NCF_MODEL_PATH = os.path.join(path, "ncf", "experiment_7/model.keras")

encoders = joblib.load(os.path.join(path, "ncf", "experiment_7", "encoders", "user_and_movies_encoders.pkl"))
user_encoder = encoders["user_encoder"]

scaler = joblib.load(os.path.join(path, "scalers", "MovieID", "experiment_3", "minMaxScalerscaler.pkl"))

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


def prepare_ncf_inputs(user: User, movies_df: pd.DataFrame):
    user_key = f"{user.gender}_{user.age}_{user.occupation}_{user.zip_code}"
    user_idx = user_encoder.transform([user_key])[0]

    user_idx_array = np.full(len(movies_df), user_idx, dtype=np.int32)

    movies_df_renamed = movies_df.rename(columns={"movie_id": "MovieID"})
    movie_idx_array = scaler.transform(movies_df_renamed[["MovieID"]].astype(np.float32)).flatten()

    return user_idx_array, movie_idx_array


def predict_ncf(user_idx_array: np.ndarray, movie_idx_array: np.ndarray) -> np.ndarray:
    preds = ncf_model.predict([user_idx_array, movie_idx_array], verbose=0)
    return preds.flatten()
