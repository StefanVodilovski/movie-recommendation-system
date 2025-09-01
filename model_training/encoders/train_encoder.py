import os

import joblib
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


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
    else:
        raise ValueError("Unsupported encoder type. Use 'label' or 'onehot'.")

    df[TARGET_COLUMN] = encoder.fit_transform(df[FEATURE_COLUMN])
    df[TARGET_COLUMN] = df[FEATURE_COLUMN]

    joblib.dump(encoder, save_path + f"/{ENCODER_TYPE}_encoder.pkl")


if __name__ == "__main__":
    DATASET_PATH = "../../datasets/full_test_dataset.csv"
    SAVE_MODEL_PATH = "../../experiments/encoders"
    FEATURE_COLUMN = "original_language"
    TARGET_COLUMN = "original_language_encoded"
    DATASET_TYPE = "csv"
    ENCODER_TYPE = "label"
    df = import_df(DATASET_TYPE, DATASET_PATH)
    df[FEATURE_COLUMN].dropna(inplace=True)
    save_path = form_save_path()
    train_encoder(save_path)
    print(f"Encoder trained and saved at {save_path}")
