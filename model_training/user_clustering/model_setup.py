# pip install pandas scikit-learn numpy
import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from configuration import BASE_PATH, numeric_cols, onehot_cols
from ExperimentConfig import ExperimentConfig


def read_dataframe(path: str) -> pd.DataFrame:
    print(f"reading dataframe from path: {path}")
    df = pd.read_csv(path)
    df = df.dropna(how="any")
    return df


def save_encoder(save_path: str, column: str, encoder) -> None:
    encoders_dir = os.path.join(save_path, "encoders")
    os.makedirs(encoders_dir, exist_ok=True)

    # Save the encoder
    encoder_file = os.path.join(encoders_dir, f"{column}_labelencoder.pkl")
    joblib.dump(encoder, encoder_file)
    print(f"Label encoder saved to {encoder_file}")


def encode_labels(
    df: pd.DataFrame,
    target_columns: list[str],
    encoded_column_names: list[str],
    save_path: str,
):
    print("Encoding labels...")
    le = LabelEncoder()
    for i, column in enumerate(target_columns):
        df[encoded_column_names[i]] = le.fit_transform(df[column])
        save_encoder(save_path, column, le)
        df = df.drop(columns=[column])
    return df


def scale_numeric_columns(
    df: pd.DataFrame, numeric_columns: list[str], save_path: str
) -> pd.DataFrame:
    print("Scaling numeric columns...")
    scaler = StandardScaler()
    for column in numeric_columns:
        df[column] = scaler.fit_transform(df[column])
        scalars_dir = os.path.join(save_path, "scalaras")
        os.makedirs(scalars_dir, exist_ok=True)
        scaler_file = os.path.join(save_path, f"{column}_numeric_columns_scaler.pkl")
        joblib.dump(scaler, scaler_file)
        print(f"Numeric column scaler saved to {scaler_file}")
    return df


def data_preprocessing(df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    print("Preprocessing data...")
    df = encode_labels(
        df,
        ["Zip-code", "Occupation"],
        ["occupation_encoded", "zipcode_endcoded"],
        save_path,
    )
    df = pd.get_dummies(
        df,
        columns=["Gender"],
        dtype=np.float32,
    )
    df = scale_numeric_columns(df, numeric_cols, save_path)
    return df


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


def plot_clustering_results(
    save_path: str,
    K_range: range,
    inertia_list: list[float],
    silhouette_list: list[float],
) -> None:
    # Plot Elbow Method
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertia_list, marker="o")
    plt.title("Elbow Method (Inertia)")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")

    # Plot Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_list, marker="o")
    plt.title("Silhouette Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig(f"{save_path}/clustering_analysis.png", dpi=300)
    plt.close()


def save_best_model(
    excperiment_config: ExperimentConfig,
    save_path: str,
    K_range: range,
    silhouette_list: list[float],
) -> None:
    print("Selecting best model based on silhouette score...")
    best_k = K_range[silhouette_list.index(max(silhouette_list))]  # highest silhouette
    kmeans_final = KMeans(
        n_clusters=best_k,
        init=excperiment_config.k_means_init,
        n_init=excperiment_config.n_init,
        random_state=excperiment_config.random_state,
    )
    kmeans_final.fit(df)
    model_save_path = save_path / f"kmeans_users_{best_k}_clusters.joblib"
    joblib.dump(kmeans_final, model_save_path)

    print(f"Best K found: {best_k} with Silhouette Score: {max(silhouette_list)}")
    df["cluster"] = kmeans_final.labels_
    datasets_path = "../../datasets/"
    df.to_csv(datasets_path / "users_with_clusters.csv", index=False)


def train_and_save_k_means(
    excperiment_config: ExperimentConfig, df: pd.DataFrame, save_path: str
) -> None:
    print("Training K-Means clustering...")
    inertia_list = []
    silhouette_list = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(
            n_clusters=k,
            init=excperiment_config.k_means_init,
            n_init=excperiment_config.n_init,
            random_state=excperiment_config.random_state,
        )
        kmeans.fit(df)
        inertia_list.append(kmeans.inertia_)
        silhouette_list.append(silhouette_score(df, kmeans.labels_))
        print(
            f"K-Means with k={k}: Inertia={kmeans.inertia_}, Silhouette Score={silhouette_list[-1]}"
        )

    plot_clustering_results(save_path, K_range, inertia_list, silhouette_list)
    save_best_model(excperiment_config, save_path, K_range, silhouette_list)


if __name__ == "__main__":
    path = "../../datasets/train_users.csv"
    save_path = form_save_path()
    df = read_dataframe(path)

    df = data_preprocessing(df, save_path)
    df = df[numeric_cols + onehot_cols]
    experiment_config = ExperimentConfig(
        dataset_name="train_users.csv",
        k_means_init="k-means++",
        n_init=10,
        random_state=42,
        columns=numeric_cols + onehot_cols,
    )
    train_and_save_k_means(experiment_config, df, save_path)
