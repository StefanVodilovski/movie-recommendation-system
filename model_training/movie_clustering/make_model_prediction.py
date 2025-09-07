import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DATASET_PATH = "../../datasets/full_train_dataset_with_embeddings.parquet"
ENCODER_PATH = (
    "../../experiments/encoders/original_language/experiment_1/label_encoder.pkl"
)
PCA_MODEL_PATH = "../../experiments/movie_clustering/experiment_2/pca/pca.pkl"
SCALER_PATH = (
    "../../experiments/movie_clustering/experiment_3/scalars/feature_matrix_scaler.pkl"
)
TIMESTAMP_SCALER_PATH = (
    "../../experiments/scalers/Timestamp/experiment_2/scalar_encoder.pkl"
)
CLUSTERING_MODEL_PATH = (
    "../../experiments/movie_clustering/experiment_3/kmeans_users_14_clusters.joblib"
)

METADATA_COLUMNS = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "adult",
    "original_language_encoded",
    "popularity",
    "vote_average",
    "vote_count",
    "timestamp_encoded",
]

PLOT_PATH = "../../experiments/movie_clustering/experiment_2/clustering_plot.png"

SAVE_DATAFRAME_PATH = "../../datasets/clustered_movies_with_embeddings.parquet"
USER_COLUMNS = ["Gender", "Age", "Occupation", "Zip-code"]


def reduce_embedding_size(df: pd.DataFrame) -> np.ndarray:
    print("Reducing Embedding size")
    embeddings = np.vstack(df["embedding"].values)
    pca = joblib.load(PCA_MODEL_PATH)
    reduced_embeddings = pca.transform(embeddings)
    return reduced_embeddings


def dataframe_scaler(feature_matrix: np.ndarray) -> np.ndarray:
    print("Scaling dataframe")
    scaler = joblib.load(SCALER_PATH)
    scaled_matrix = scaler.transform(feature_matrix)
    return scaled_matrix


def prepare_data(df: pd.DataFrame) -> np.ndarray:
    print("Preparing data...")
    df = df.dropna().copy()
    encoder = joblib.load(ENCODER_PATH)
    df["original_language_encoded"] = encoder.transform(df["original_language"])
    scaler = joblib.load(TIMESTAMP_SCALER_PATH)
    df["timestamp_encoded"] = scaler.transform(df[["Timestamp"]]).flatten()
    reduced_embeddings = reduce_embedding_size(df)
    metadata_df = df[METADATA_COLUMNS]
    feature_matrix = np.hstack([reduced_embeddings, metadata_df])
    scaled_matrix = dataframe_scaler(feature_matrix)
    return scaled_matrix, df


def cluster_movies(df: pd.DataFrame, scaled_matrix: np.ndarray):
    print("Clustering movies...")
    clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
    cluster_labels = clustering_model.predict(scaled_matrix)
    df["cluster"] = cluster_labels
    return df


def plot_clusters(df: pd.DataFrame, scaled_matrix: np.ndarray):
    print(f"Saving cluster plot to {PLOT_PATH}...")
    pca_vis = PCA(n_components=2)
    reduced_2d = pca_vis.fit_transform(scaled_matrix)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        reduced_2d[:, 0], reduced_2d[:, 1], c=df["cluster"], cmap="tab20", alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title("Movie Clusters (2D PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def save_results(df: pd.DataFrame):
    print(f"Saving clustered movies to {SAVE_DATAFRAME_PATH}...")
    os.makedirs(os.path.dirname(SAVE_DATAFRAME_PATH), exist_ok=True)
    # Keep metadata + cluster
    result_df = df[USER_COLUMNS + ["cluster"]].copy()
    result_df.to_parquet(SAVE_DATAFRAME_PATH, index=False)


if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    scaled_matrix, df = prepare_data(df)
    df = cluster_movies(df, scaled_matrix)
    plot_clusters(df, scaled_matrix)
    save_results(df)
