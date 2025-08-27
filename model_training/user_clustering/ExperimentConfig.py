from pydantic import BaseModel


numeric_cols = ["Age", "occupation_encoded", "Zipcode_endcoded"]
onehot_cols = ["Gender_M", "Gender_F"]


class ExperimentConfig(BaseModel):
    dataset_name: str = "train_users.csv"
    k_means_init: str = "k-means++"
    n_init: int = 10
    random_state: int = 42
    columns: list[str] = numeric_cols + onehot_cols
