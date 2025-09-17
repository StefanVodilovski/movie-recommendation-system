import os

import joblib
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def import_df(type: str, path: str) -> pd.DataFrame:
    if type == "csv":
        df = pd.read_csv(path)
    elif type == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file type. Use 'csv' or 'parquet'.")
    return df


def form_save_path() -> str:
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    experiment_count = sum(
        os.path.isdir(os.path.join(SAVE_MODEL_PATH, entry))
        for entry in os.listdir(SAVE_MODEL_PATH)
    )
    res = f"{SAVE_MODEL_PATH}/{FEATURE_COLUMN}/experiment_{experiment_count + 1}"
    os.makedirs(res, exist_ok=True)
    print(f"Save path formed: {res}")
    return res


def train_encoder(save_path: str) -> None:
    if ENCODER_TYPE == "label":
        encoder = LabelEncoder()
    elif ENCODER_TYPE == "onehot":
        encoder = OneHotEncoder()
    elif ENCODER_TYPE == "scalar":
        encoder = StandardScaler()
    elif ENCODER_TYPE == "minMaxScaler":
        encoder = MinMaxScaler()
    else:
        raise ValueError("Unsupported encoder type. Use 'label' or 'onehot'.")

    df[TARGET_COLUMN] = encoder.fit_transform(df[[FEATURE_COLUMN]])
    joblib.dump(encoder, save_path + f"/{ENCODER_TYPE}scaler.pkl")


if __name__ == "__main__":
    DATASET_PATH = "../../datasets/full_train_dataset_with_embeddings.parquet"
    SAVE_MODEL_PATH = "../../experiments/scalers"
    FEATURE_COLUMN = "MovieID"
    TARGET_COLUMN = "movie_id_scaled"

    DATASET_TYPE = "parquet"
    ENCODER_TYPE = "minMaxScaler"
    df = import_df(DATASET_TYPE, DATASET_PATH)
    df.dropna(inplace=True)
    if ENCODER_TYPE == "scalar" and FEATURE_COLUMN == "Timestamp":
        df[FEATURE_COLUMN] = pd.to_datetime(df[FEATURE_COLUMN], errors="coerce")
        df[FEATURE_COLUMN] = df[FEATURE_COLUMN].dt.year
    save_path = form_save_path()
    train_encoder(save_path)
    print(f"Encoder trained and saved at {save_path}")
