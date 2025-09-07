from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    experiment_number: int
    epochs: int
    batch_size: int
    verbose: int
    mf_dim: int
    model_layers: list
    mlp_reg_layers: list
    embeddings_initializer: str
    embeddings_regularizer: str | None
    mf_regularization: float
