BASE_PATH = "../../experiments/movie_clustering"
LABEL_ENCODER_PATH = (
    "../../experiments/encoders/original_language/experiment_1/label_encoder.pkl"
)
TIMESTAMP_SCALER_PATH = (
    "../../experiments/scalers/Timestamp/experiment_2/scalar_encoder.pkl"
)

metadata_columns = [
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
embedding_column = ["embedding"]
