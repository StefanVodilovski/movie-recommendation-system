from pydantic import BaseModel


class Vector(BaseModel):
    columns: list[str]


class ExperimentConfig(BaseModel):
    dataset_name: str = "full_train_dataset_with_embeddings.parquet"
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 256
    learning_rate: float = 0.003
    optimizer: str = "Adam"
    loss_function: str = "mse"
    num_epochs: int = 50
    validation_split: float = 0.2
    input_vectors: list[Vector]
    target_column: str = "Rating"


genre_cols = [
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
]
numeric_cols = ["popularity", "vote_average", "vote_count", "original_langage_encoded"]
user_cols = [
    "UserID",
    "Age",
    "occupation_encoded",
    "Zipcode_endcoded",
    "Gender_M",
    "Gender_F",
]
