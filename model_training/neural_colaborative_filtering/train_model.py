from ast import Lambda
import json
import joblib
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from ExperimentConfig import ExperimentConfig
import os

import torch
from torchmetrics.retrieval import RetrievalRecall, RetrievalNormalizedDCG


BASE_PATH = "../../experiments/ncf_model"
MF_DIM = 128


def form_save_path() -> tuple[str, int]:
    print("Forming Save path")
    os.makedirs(BASE_PATH, exist_ok=True)
    experiment_count = sum(
        os.path.isdir(os.path.join(BASE_PATH, entry)) for entry in os.listdir(BASE_PATH)
    )
    res = f"{BASE_PATH}/experiment_{experiment_count + 1}"
    os.makedirs(res, exist_ok=True)
    print(f"Save path formed: {res}")
    return res, experiment_count + 1


def create_ids(
    df: pd.DataFrame, save_path: str
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    df["user_key"] = (
        df["Gender"].astype(str)
        + "_"
        + df["Age"].astype(str)
        + "_"
        + df["Occupation"].astype(str)
        + "_"
        + df["Zip-code"].astype(str)
    )

    user_encoder = LabelEncoder()
    df["user_id"] = user_encoder.fit_transform(df["user_key"])
    scaler = joblib.load(
        f"../../experiments/scalers/MovieID/experiment_3/minMaxScalerscaler.pkl"
    )

    df["item_id"] = scaler.transform(df[["MovieID"]])
    return df, user_encoder, scaler


def mf_slice(x):
    x = tf.squeeze(x, [1])
    return x[:, :MF_DIM]


def mlp_slice(x):
    x = tf.squeeze(x, [1])
    return x[:, :MF_DIM]


def construct_model(
    num_users: int, num_items: int, experiment_config: ExperimentConfig
):
    user_input = tf.keras.layers.Input(shape=(1,), dtype="int32", name="user_input")
    item_input = tf.keras.layers.Input(shape=(1,), dtype="int32", name="item_input")

    # Embeddings
    user_embeddings = tf.keras.layers.Embedding(
        num_users,
        experiment_config.mf_dim + experiment_config.model_layers[0] // 2,
        embeddings_initializer=experiment_config.embeddings_initializer,
        embeddings_regularizer=experiment_config.embeddings_regularizer,
        input_length=1,
        name="embedding_user",
    )(user_input)

    item_embeddings = tf.keras.layers.Embedding(
        num_items,
        experiment_config.mf_dim + experiment_config.model_layers[0] // 2,
        embeddings_initializer=experiment_config.embeddings_initializer,
        embeddings_regularizer=experiment_config.embeddings_regularizer,
        input_length=1,
        name="embedding_item",
    )(item_input)

    # MF part
    mf_user_latent = tf.keras.layers.Lambda(mf_slice, name="embedding_user_mf")(
        user_embeddings
    )
    mf_item_latent = tf.keras.layers.Lambda(mf_slice, name="embedding_item_mf")(
        item_embeddings
    )
    mf_vector = tf.keras.layers.Multiply()([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_user_latent = tf.keras.layers.Lambda(mlp_slice, name="embedding_user_mlp")(
        user_embeddings
    )
    mlp_item_latent = tf.keras.layers.Lambda(mlp_slice, name="embedding_item_mlp")(
        item_embeddings
    )
    mlp_vector = tf.keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])

    num_layer = len(experiment_config.model_layers)
    for layer in range(1, num_layer):
        mlp_vector = tf.keras.layers.Dense(
            experiment_config.model_layers[layer],
            activation="relu",
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            kernel_regularizer=tf.keras.regularizers.l2(
                experiment_config.mlp_reg_layers[layer]
            ),
        )(mlp_vector)

    # Combine MF + MLP
    predict_vector = tf.keras.layers.Concatenate()([mf_vector, mlp_vector])

    # Prediction layer
    logits = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer="lecun_uniform",
        name="rating_prediction",
    )(predict_vector)

    model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=logits)
    return model


def save_model(
    model,
    experiment_config: ExperimentConfig,
    base_path: str,
    user_encoder,
    item_encoder,
):
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

    encoders_dir = os.path.join(base_path, "encoders")
    os.makedirs(encoders_dir, exist_ok=True)
    encoder_file = os.path.join(encoders_dir, f"user_and_movies_encoders.pkl")
    joblib.dump(
        {"user_encoder": user_encoder, "item_encoder": item_encoder},
        encoder_file,
    )

    print("Model and config saved Successfully.")


def train_ncf(df: pd.DataFrame, experiment_config: ExperimentConfig, save_path: str):
    df, user_encoder, item_encoder = create_ids(df, save_path)
    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()

    model = construct_model(num_users, num_items, experiment_config)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_users = train["user_id"].values
    train_items = train["item_id"].values
    train_ratings = train["Rating"].values

    test_users = test["user_id"].values
    test_items = test["item_id"].values
    test_ratings = test["Rating"].values

    history = model.fit(
        [train_users, train_items],
        train_ratings,
        validation_data=([test_users, test_items], test_ratings),
        epochs=experiment_config.epochs,
        batch_size=experiment_config.batch_size,
        verbose=experiment_config.verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=5,
                restore_best_weights=True,
                mode="min",
            )
        ],
    )
    test_loss, test_mae = model.evaluate(
        [test_users, test_items], test_ratings, verbose=1
    )
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    save_model(model, experiment_config, save_path, user_encoder, item_encoder)

    return model, history, user_encoder, item_encoder


def validate_and_save_results(model_history, filename: str):
    print("Validating and saving results...")
    train_mae = model_history.history["mae"]
    val_mae = model_history.history["val_mae"]
    epochs = range(1, len(train_mae) + 1)

    plt.figure()
    plt.plot(epochs, train_mae, label="Train MAE")
    plt.plot(epochs, val_mae, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training vs. Validation MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # high resolution
    plt.close()  # close the figure to free memory
    print(f"Train MAE: {train_mae}, validation MAE: {val_mae}")
    print("Results validated and saved.")


def evaluate_with_torchmetrics(
    model,
    test_df: pd.DataFrame,
    save_path: str,
    k: int = 10,
    threshold: float = 4.0,
):
    """
    Evaluate Recall@K and NDCG@K using TorchMetrics.
    """
    print("Evaluating ranking metrics with TorchMetrics...")

    # Extract arrays
    u_test = test_df["user_id"].values
    i_test = test_df["item_id"].values
    y_test = test_df["Rating"].values

    # Predict with Keras model
    y_pred = model.predict([u_test, i_test], batch_size=512).flatten()

    # Binary relevance (1 if rating >= threshold else 0)
    y_true_binary = (y_test >= threshold).astype(int)

    # Convert to torch tensors
    preds = torch.tensor(y_pred, dtype=torch.float32)
    target = torch.tensor(y_true_binary, dtype=torch.int32)
    indexes = torch.tensor(u_test, dtype=torch.long)  # must be long!

    # Define metrics
    recall_metric = RetrievalRecall(top_k=k)
    ndcg_metric = RetrievalNormalizedDCG(top_k=k)

    recall10 = recall_metric(preds, target, indexes=indexes).item()
    ndcg10 = ndcg_metric(preds, target, indexes=indexes).item()

    evaluation = {"Recall@10": recall10, "NDCG@10": ndcg10}
    with open(f"{save_path}/evaluation_metrics.json", "w") as f:
        json.dump(evaluation, f)

    print(f"Recall@{k}: {recall10:.4f}")
    print(f"NDCG@{k}: {ndcg10:.4f}")
    return recall10, ndcg10


if __name__ == "__main__":
    df = pd.read_parquet("../../datasets/full_train_dataset_with_embeddings.parquet")

    base_save_path, experiment_number = form_save_path()
    experiment_config = ExperimentConfig(
        experiment_number=experiment_number,
        epochs=250,
        batch_size=512,
        verbose=1,
        mf_dim=128,
        model_layers=[64, 32, 16, 8],
        mlp_reg_layers=[0, 0, 0, 0],
        embeddings_initializer="uniform",
        embeddings_regularizer="l2",
        mf_regularization=1e-4,
    )
    model, history, user_encoder, item_encoder = train_ncf(
        df, experiment_config, base_save_path
    )
    validate_and_save_results(history, f"{base_save_path}/training_validation_rmse.png")
    _, test = train_test_split(df, test_size=0.2, random_state=42)
    recall, ndcg = evaluate_with_torchmetrics(
        model, test, base_save_path, k=10, threshold=4.0
    )
