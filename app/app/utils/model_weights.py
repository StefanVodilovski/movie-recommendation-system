def get_model_weights(num_ratings: int) -> tuple[float, float]:
    if num_ratings <= 3:
        return 1.0, 0.0
    elif num_ratings <= 8:
        return 0.8, 0.2
    elif num_ratings <= 15:
        return 0.5, 0.5
    elif num_ratings <= 30:
        return 0.3, 0.7
    else:
        return 0.2, 0.8
