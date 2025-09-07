import json
import os
import joblib
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
import tensorflow as tf
from torchmetrics.retrieval import RetrievalRecall, RetrievalNormalizedDCG

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Concatenate,
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import torch
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
    for i, column in enumerate(target_columns):
        le = joblib.load(
            f"../../experiments/encoders/{column}/experiment_3/label_encoder.pkl"
        )
        df[encoded_column_names[i]] = le.transform(df[column])
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
    # Inputs (dense user features, movie features, overview embedding)
    inp_user = Input(shape=(user_dim,), name="user_id")
    inp_movie = Input(shape=(movie_dim,), name="movie_feat")
    inp_overview = Input(shape=(overview_dim,), name="overview_emb_feat")

    # User tower (dense features)
    u = Dense(128, activation="relu")(inp_user)
    u = BatchNormalization()(u)
    u = Dropout(0.2)(u)
    u = Dense(64, activation="relu")(u)
    u = Dropout(0.2)(u)

    # Movie tower
    m = Dense(256, activation="relu")(inp_movie)
    m = BatchNormalization()(m)
    m = Dropout(0.3)(m)
    m = Dense(128, activation="relu")(m)
    m = Dropout(0.2)(m)

    # Overview tower
    e = Dense(256, activation="relu")(inp_overview)
    e = BatchNormalization()(e)
    e = Dropout(0.3)(e)
    e = Dense(128, activation="relu")(e)
    e = Dropout(0.2)(e)

    # Combine
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

    out = Dense(1, name="rating")(x)

    model = Model([inp_user, inp_movie, inp_overview], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(experiment_config.learning_rate),
        loss=experiment_config.loss_function,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def validate_and_save_results(model_history, filename: str):
    print("Validating and saving results...")
    # metric name 'rmse' as compiled above
    train_rmse = model_history.history.get("rmse", [])
    val_rmse = model_history.history.get("val_rmse", [])
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
    plt.close()
    print(f"Train RMSE: {train_rmse}, validation RMSE: {val_rmse}")
    print("Results validated and saved.")


def save_model_and_config(base_path: str, model, experiment_config: ExperimentConfig):
    print("Saving model and config...")
    model.save(f"{base_path}/model.keras")
    model_json = model.to_json()
    with open(f"{base_path}/model_architecture.json", "w") as f:
        f.write(model_json)
    model.save_weights(f"{base_path}/model.weights.h5")
    with open(os.path.join(base_path, "experiment_config.json"), "w") as f:
        json.dump(experiment_config.model_dump(), f, indent=2)
    print("Model and config saved Successfully.")


def form_experiment_config() -> ExperimentConfig:
    experiment_config = ExperimentConfig(
        dataset_name="datasets/train_df_with_left_out_user.parquet",
        test_size=0,
        random_state=0,
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


# ----- NEW: build item feature map from combined train+test -----
def build_item_feature_map(df_all, item_id_col="MovieID"):
    """
    returns:
      item_ids: list of unique item ids
      item_feat_map: dict[item_id] = (movie_feat_vector, overview_vector)
    """
    item_rows = df_all.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)
    item_ids = item_rows[item_id_col].to_numpy()
    item_feat_map = {}
    for _, row in item_rows.iterrows():
        item_id = row[item_id_col]
        movie_feat = row[genre_cols + numeric_cols].to_numpy(dtype=np.float32)
        overview_feat = np.array(row["embedding"], dtype=np.float32)
        item_feat_map[item_id] = (movie_feat, overview_feat)
    return item_ids.tolist(), item_feat_map


# ----- NEW: proper leave-one-out evaluation (score against full catalog minus train items) -----
def evaluate_ranking_leave_one_out(model, train_df, test_df, k=10):
    """
    train_df: DataFrame with user's training interactions
    test_df: DataFrame with exactly one left-out test interaction per user
    Model expects inputs: [user_vecs, movie_vecs, overview_vecs]
    """
    print("Evaluating ranking metrics with full-catalog scoring...")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    all_item_ids, item_feat_map = build_item_feature_map(
        combined_df, item_id_col="MovieID"
    )

    # build per-user train item sets to exclude from candidate list
    user_train_items = defaultdict(set)
    for _, r in train_df.iterrows():
        user_train_items[r["UserID"]].add(r["MovieID"])

    recalls = []
    ndcgs = []

    # iterate test rows (one left-out per user)
    for idx, test_row in test_df.iterrows():
        uid = test_row["UserID"]
        leftout_item = test_row["MovieID"]
        leftout_rating = float(test_row[TARGET_COLUMN])

        # candidate items = all items minus user's train items
        candidate_items = [
            it for it in all_item_ids if it not in user_train_items.get(uid, set())
        ]
        if len(candidate_items) == 0:
            continue

        # build arrays for prediction
        # user vector from test row (same features as used in training)
        user_vector = test_row[user_cols].to_numpy(
            dtype=np.float32
        )  # shape (user_dim,)
        user_repeat = np.repeat(
            user_vector[np.newaxis, :], len(candidate_items), axis=0
        )

        movie_feats = np.stack([item_feat_map[it][0] for it in candidate_items]).astype(
            np.float32
        )
        overview_feats = np.stack(
            [item_feat_map[it][1] for it in candidate_items]
        ).astype(np.float32)

        # predict scores for all candidates
        scores = model.predict(
            [user_repeat, movie_feats, overview_feats], batch_size=1024, verbose=0
        ).flatten()

        # find index of left-out item among candidates (skip if missing)
        try:
            idx_left = candidate_items.index(leftout_item)
        except ValueError:
            # left-out item not present in combined item map, skip
            continue

        ranked_idx = np.argsort(-scores)
        top_k_idx = ranked_idx[:k]

        # For leave-one-out there's a single relevant item: compute recall (1 if left-out is in top-k)
        recall = 1.0 if idx_left in top_k_idx else 0.0
        recalls.append(recall)

        # NDCG: compute contribution of left-out item if in top-k, else 0.
        if idx_left in top_k_idx:
            rank_pos = int(np.where(ranked_idx == idx_left)[0][0])  # 0-based
            gain = (2**leftout_rating - 1) / math.log2(rank_pos + 2)
            idcg = (2**leftout_rating - 1) / math.log2(1 + 1)
            ndcgs.append(gain / idcg)
        else:
            ndcgs.append(0.0)

    recall_mean = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    ndcg_mean = float(np.mean(ndcgs)) if len(ndcgs) > 0 else 0.0

    return recall_mean, ndcg_mean


def evaluate_model_and_save(model, save_path, train_df, test_df):
    recall10, ndcg10 = evaluate_ranking_leave_one_out(model, train_df, test_df, k=10)
    evaluation = {"Recall@10": recall10, "NDCG@10": ndcg10}
    with open(f"{save_path}/evaluation_metrics.json", "w") as f:
        json.dump(evaluation, f)
    print(f"Recall@10: {recall10:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")


def evaluate_model(
    model, save_path, u_test, i_test, e_test, y_test, uid_test, k=10, threshold=4.0
):
    print("Evaluating ranking metrics...")

    # Get predictions
    y_pred = model.predict([u_test, i_test, e_test], batch_size=512).flatten()

    # Build binary relevance (1 if rating >= threshold, else 0)
    y_true_binary = (y_test >= threshold).astype(int)

    # Convert to torch tensors
    preds = torch.tensor(y_pred, dtype=torch.float32)
    target = torch.tensor(y_true_binary, dtype=torch.int32)
    indexes = torch.tensor(uid_test, dtype=torch.long)

    # Metrics
    recall_metric = RetrievalRecall(top_k=k)
    ndcg_metric = RetrievalNormalizedDCG(top_k=k)

    recall10 = recall_metric(preds, target, indexes=indexes).item()
    ndcg10 = ndcg_metric(preds, target, indexes=indexes).item()

    evaluation = {"Recall@10": recall10, "NDCG@10": ndcg10}
    with open(f"{save_path}/evaluation_metrics.json", "w") as f:
        json.dump(evaluation, f)

    print(f"Recall@10: {recall10:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")


if __name__ == "__main__":
    train_path = "../../datasets/train_df_with_left_out_user.parquet"
    test_path = "../../datasets/test_df_with_left_out_user.parquet"
    train_df = read_parquet_dataframe(train_path)
    test_df = read_parquet_dataframe(test_path)
    train_df = data_preprocessing(train_df)
    test_df = data_preprocessing(test_df)
    experiment_config = form_experiment_config()

    # build training arrays (used by model.fit)
    u_train, i_train, e_train = get_input_vectors(train_df)
    y_train = train_df[experiment_config.target_column].to_numpy(dtype=np.float32)

    u_test, i_test, e_test = get_input_vectors(test_df)
    y_test = test_df[experiment_config.target_column].to_numpy(dtype=np.float32)

    n_users = train_df.UserID.max() + 1
    user_dim = u_train.shape[1]
    overview_dim = e_train.shape[1]
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
                monitor="val_rmse",
                patience=5,
                restore_best_weights=True,
            )
        ],
    )

    save_path = form_save_path()
    save_model_and_config(save_path, model, experiment_config)
    validate_and_save_results(history, save_path + "/training_results.png")

    # Evaluate ranking properly (score left-out item among full candidate catalog)
    # evaluate_model_and_save(model, save_path, train_df, test_df)
    uid_test = test_df["UserID"].to_numpy()
    evaluate_model(
        model, save_path, u_test, i_test, e_test, y_test, uid_test, k=10, threshold=4.0
    )
