from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    dataset_name: str = "train_users.csv"
    k_means_init: str = "k-means++"
    n_init: int = 10
    random_state: int = 42
    metadata_columns: list[str]
    embedding_column: list[str] = ["embedding"]
    target: str = "cluster"
