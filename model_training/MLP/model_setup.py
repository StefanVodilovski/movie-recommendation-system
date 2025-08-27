import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Concatenate,
)
from tensorflow.keras.models import Model
from sklearn.calibration import LabelEncoder
import matplotlib.pyplot as plt
from ExperimentConfig import ExperimentConfig, Vector
from configuration import BASE_PATH, TARGET_COLUMN, genre_cols, numeric_cols, user_cols


def read_parquet_dataframe(path: str) -> pd.DataFrame:
    print(f"reading dataframe from path: {path}")
    df = pd.read_parquet(path)
    df = df.dropna(how="any")
    return df


def encode_labels(
    df: pd.DataFrame, target_columns: list[str], encoded_column_names: list[str]
):
    print("Encoding labels...")
    le = LabelEncoder()
    for i, column in enumerate(target_columns):
        df[encoded_column_names[i]] = le.fit_transform(df[column])
        df = df.drop(columns=[column])
    return df


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing data...")
    target_columns = ["Zip-code", "original_language", "Occupation"]
    encoded_columns = [
        "Zipcode_endcoded",
        "original_langage_encoded",
        "occupation_encoded",
    ]
    df = encode_labels(df, target_columns, encoded_columns)
    df = pd.get_dummies(
        df,
        columns=["Gender"],
        dtype=np.float32,
    )

    return df


def get_input_vectors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("Forming input vectors...")
    user_vecs = df[user_cols].to_numpy(dtype=np.float32)
    item_vecs = df[genre_cols + numeric_cols].to_numpy(dtype=np.float32)
    overview_embdeddings_vecs = np.stack(df["embedding"].values).astype(np.float32)
    return user_vecs, item_vecs, overview_embdeddings_vecs


def form_model_architecture(
    experiment_config: ExperimentConfig, n_users, user_dim, movie_dim, overview_dim
):
    print("Forming Model architecture...")
    # ——— Inputs ———
    inp_user = Input(shape=(user_dim,), name="user_id")
    inp_movie = Input(shape=(movie_dim,), name="movie_feat")
    inp_overview = Input(shape=(overview_dim,), name="overview_emb_feat")
    # ——— User tower ———
    u = Embedding(input_dim=n_users, output_dim=user_dim, name="user_emb")(inp_user)
    u = Flatten()(u)
    u = Dense(64, activation="relu")(u)
    u = Dropout(0.2)(u)

    # ——— Movie tower ———
    m = Dense(256, activation="relu")(inp_movie)
    m = BatchNormalization()(m)
    m = Dropout(0.3)(m)
    m = Dense(128, activation="relu")(m)
    m = Dropout(0.2)(m)

    # ——— Overview tower ———
    e = Dense(256, activation="relu")(inp_overview)
    e = BatchNormalization()(e)
    e = Dropout(0.3)(e)
    e = Dense(128, activation="relu")(e)
    e = Dropout(0.2)(e)

    # ——— Combine & head ———
    x = Concatenate()([u, m, e])
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    out = Dense(1, name="rating")(x)  # linear activation for regression

    model = Model([inp_user, inp_movie, inp_overview], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(experiment_config.learning_rate),
        loss=experiment_config.loss_function,
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


def validate_and_save_results(model_history, filename: str):
    print("Validating and saving results...")
    train_rmse = model_history.history["root_mean_squared_error"]
    val_rmse = model_history.history["val_root_mean_squared_error"]
    epochs = range(1, len(train_rmse) + 1)

    plt.figure()
    plt.plot(epochs, train_rmse, label="Train RMSE")
    plt.plot(epochs, val_rmse, label="Val   RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training vs. Validation RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # high resolution
    plt.close()  # close the figure to free memory
    print(f"Train RMSE: {train_rmse}, validation RMSE: {val_rmse}")
    print("Results validated and saved.")


def save_model_and_config(base_path: str, model, experiment_config: ExperimentConfig):
    print("Saving model and config...")
    model.save(
        f"{base_path}/model.keras",
    )

    model_json = model.to_json()
    with open(f"{base_path}/model_architecture.json", "w") as f:
        f.write(model_json)

    model.save_weights(f"{base_path}/model.weights.h5")
    with open(os.path.join(base_path, "experiment_config.json"), "w") as f:
        json.dump(experiment_config.model_dump(), f, indent=2)
    print("Model and config saved Successfully.")


def form_experiment_config() -> ExperimentConfig:
    experiment_config = ExperimentConfig(
        dataset_name="full_train_dataset_with_embeddings.parquet",
        test_size=0.2,
        random_state=42,
        batch_size=256,
        learning_rate=1e-3,
        optimizer="Adam",
        loss_function="mse",
        num_epochs=15,
        validation_split=0.2,
        input_vectors=[
            Vector(columns=user_cols),
            Vector(columns=genre_cols),
            Vector(columns=numeric_cols),
        ],
        target_column=TARGET_COLUMN,
    )

    return experiment_config


def form_save_path() -> str:
    print("Forming Save path")
    os.makedirs(BASE_PATH, exist_ok=True)
    experiment_count = sum(
        os.path.isdir(os.path.join(BASE_PATH, entry)) for entry in os.listdir(BASE_PATH)
    )
    res = f"{BASE_PATH}/experiment_{experiment_count + 1}"
    os.makedirs(res, exist_ok=True)
    print(f"Save path formed: {res}")
    return res


if __name__ == "__main__":
    path = "../../datasets/full_train_dataset_with_embeddings.parquet"
    df = read_parquet_dataframe(path)
    df = data_preprocessing(df)
    experiment_config = form_experiment_config()
    user_vecs, item_vecs, overview_embdeddings_vecs = get_input_vectors(df)

    ratings = df[experiment_config.target_column].to_numpy(dtype=np.float32)

    u_train, u_test, e_train, e_test, i_train, i_test, y_train, y_test = (
        train_test_split(
            user_vecs,
            overview_embdeddings_vecs,
            item_vecs,
            ratings,
            test_size=experiment_config.test_size,
            random_state=experiment_config.random_state,
        )
    )

    n_users = df.UserID.max() + 1  # replace with: df.UserID.max()+1
    user_dim = u_train.shape[1]

    overview_dim = overview_embdeddings_vecs.shape[1]  # embedding dimension
    n_genres = len(genre_cols)
    n_numeric = len(numeric_cols)
    movie_dim = n_genres + n_numeric

    model = form_model_architecture(
        experiment_config, n_users, user_dim, movie_dim, overview_dim
    )

    model.summary()

    print("Started model training...")
    history = model.fit(
        x=[u_train, i_train, e_train],
        y=y_train,
        batch_size=experiment_config.batch_size,
        epochs=experiment_config.num_epochs,
        validation_split=experiment_config.validation_split,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_root_mean_squared_error",
                patience=5,
                restore_best_weights=True,
            )
        ],
    )
    save_path = form_save_path()
    save_model_and_config(save_path, model, experiment_config)
    validate_and_save_results(history, save_path + "/training_results.png")
