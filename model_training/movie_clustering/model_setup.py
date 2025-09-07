# pip install pandas scikit-learn numpy
import json
import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from configuration import (
    BASE_PATH,
    LABEL_ENCODER_PATH,
    TIMESTAMP_SCALER_PATH,
    metadata_columns,
    embedding_column,
)
from ExperimentConfig import ExperimentConfig


def read_dataframe(path: str) -> pd.DataFrame:
    print(f"reading dataframe from path: {path}")
    df = pd.read_parquet(path)
    df = df.dropna(how="any")
    return df.sample(frac=0.1, random_state=42).reset_index(drop=True)


def encode_labels(
    df: pd.DataFrame,
    target_column: str,
    encoded_column_name: str,
):
    print("Encoding labels...")
    le = joblib.load(LABEL_ENCODER_PATH)
    df[encoded_column_name] = le.transform(df[target_column])
    df = df.drop(columns=[target_column])
    return df


def reduce_embedding_size(df: pd.DataFrame, save_path: str) -> np.ndarray:
    embeddings = np.vstack(df["embedding"].values)
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    pca_dir = os.path.join(save_path, "pca")
    os.makedirs(pca_dir, exist_ok=True)

    # Save the encoder
    pca_file = os.path.join(pca_dir, f"pca.pkl")
    joblib.dump(pca, pca_file)
    print(f"PCA Model saved to {pca_file}")
    return reduced_embeddings


def scale_timestamp(
    df: pd.DataFrame, feature_column: str, target_column: str
) -> pd.DataFrame:
    print("Scaling timestamp...")
    le = joblib.load(TIMESTAMP_SCALER_PATH)
    df[target_column] = le.transform(df[[feature_column]]).flatten()
    df = df.drop(columns=[feature_column])
    return df


def data_preprocessing(
    df: pd.DataFrame, save_path: str
) -> tuple[pd.DataFrame, np.ndarray]:
    print("Preprocessing data...")
    df = encode_labels(df, "original_language", "original_language_encoded")
    df = scale_timestamp(df, "Timestamp", "timestamp_encoded")
    reduced_embeddings = reduce_embedding_size(df, save_path)
    return df, reduced_embeddings


def dataframe_scaler(feature_matrix: np.ndarray, save_path: str) -> np.ndarray:
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix)
    scalars_dir = os.path.join(save_path, "scalars")
    os.makedirs(scalars_dir, exist_ok=True)
    scaler_file = os.path.join(scalars_dir, f"feature_matrix_scaler.pkl")
    joblib.dump(scaler, scaler_file)
    return scaled_matrix


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


def save_model_config(save_path: str, experiment_config: ExperimentConfig):
    with open(os.path.join(save_path, "experiment_config.json"), "w") as f:
        json.dump(experiment_config.model_dump(), f, indent=2)
    print("Model and config saved Successfully.")


def save_best_model_and_config(
    scaled_matrix: np.ndarray,
    experiment_config: ExperimentConfig,
    save_path: str,
    K_range: range,
    silhouette_list: list[float],
) -> None:
    print("Selecting best model based on silhouette score...")
    best_k = K_range[silhouette_list.index(max(silhouette_list))]  # highest silhouette
    kmeans_final = KMeans(
        n_clusters=best_k,
        init=experiment_config.k_means_init,
        n_init=experiment_config.n_init,
        random_state=experiment_config.random_state,
    )
    kmeans_final.fit(scaled_matrix)
    model_save_path = f"{save_path}/kmeans_users_{best_k}_clusters.joblib"
    joblib.dump(kmeans_final, model_save_path)
    print(f"Best K found: {best_k} with Silhouette Score: {max(silhouette_list)}")


def train_and_save_k_means(
    experiment_config: ExperimentConfig, scaled_matrix: np.ndarray, save_path: str
) -> None:
    print("Training K-Means clustering...")
    inertia_list = []
    silhouette_list = []
    K_range = range(2, 15)

    for k in K_range:
        kmeans = KMeans(
            n_clusters=k,
            init=experiment_config.k_means_init,
            n_init=experiment_config.n_init,
            random_state=experiment_config.random_state,
        )
        kmeans.fit(scaled_matrix)
        inertia_list.append(kmeans.inertia_)
        silhouette_list.append(silhouette_score(scaled_matrix, kmeans.labels_))
        print(
            f"K-Means with k={k}: Inertia={kmeans.inertia_}, Silhouette Score={silhouette_list[-1]}"
        )

    plot_clustering_results(save_path, K_range, inertia_list, silhouette_list)
    save_best_model_and_config(
        scaled_matrix, experiment_config, save_path, K_range, silhouette_list
    )


def plot_clusters_pca(
    experiment_config: ExperimentConfig, df: pd.DataFrame, save_path: str
):
    pca = PCA(n_components=2)
    components = pca.fit_transform(
        df[experiment_config.metadata_columns + experiment_config.embedding_column]
    )

    plt.figure(figsize=(8, 6))
    for cluster_id in df[experiment_config.target].unique():
        cluster_data = components[df[experiment_config.target] == cluster_id]
        plt.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Clusters Visualization (PCA)")
    plt.legend()
    plt.savefig(f"{save_path}/visualized_clustering.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    path = "../../datasets/full_train_dataset_with_embeddings.parquet"
    save_path = form_save_path()
    df = read_dataframe(path)

    df, reduced_embeddings = data_preprocessing(df, save_path)
    metadata_df = df[metadata_columns]
    feature_matrix = np.hstack([reduced_embeddings, metadata_df])
    scaled_matrix = dataframe_scaler(feature_matrix, save_path)
    experiment_config = ExperimentConfig(
        dataset_name="full_train_dataset_with_embeddings.parquet",
        k_means_init="k-means++",
        n_init=10,
        random_state=42,
        metadata_columns=metadata_columns,
        embedding_column=embedding_column,
        target="cluster",
    )
    train_and_save_k_means(experiment_config, scaled_matrix, save_path)
    plot_clusters_pca(experiment_config, scaled_matrix, save_path)
