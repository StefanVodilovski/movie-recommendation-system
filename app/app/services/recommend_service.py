from fastapi import Depends, HTTPException

from app.ml.mlp_predictor import predict_mlp
from app.ml.ncf_predictor import predict_ncf
from app.repository.movie_repository import MovieRepository, get_movie_repository
from app.repository.rating_repository import RatingRepository, get_rating_repository
from app.repository.recommend_repository import RecommendRepository, get_recommend_repository
from app.repository.user_repository import UserRepository, get_user_repository
from app.schemas.recommend import RecommendationSchema, RecommendationsResponse
from app.utils.model_weights import get_model_weights


class RecommendService:
    def __init__(self, recommend_repo: RecommendRepository, user_repo: UserRepository,
                 movie_repo: MovieRepository, rating_repo: RatingRepository) -> None:
        self.recommend_repo = recommend_repo
        self.user_repo = user_repo
        self.movie_repo = movie_repo
        self.rating_repo = rating_repo

    def get_cached_recommendations(self, user_id: int) -> RecommendationsResponse | None:
        cached_recs = self.recommend_repo.get_recommendations_by_user_id(user_id)
        if not cached_recs:
            return None

        recommendations = [
            RecommendationSchema(
                movie_id=rec.UserRecommendation.movie_id,
                name=rec.name,
                score=rec.UserRecommendation.score
            ) for rec in cached_recs
        ]
        return RecommendationsResponse(user_id=user_id, recommendations=recommendations)

    def get_recommendations(self, user_id: int) -> RecommendationsResponse:
        user = self.user_repo.get_user_by_id(user_id=user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        cached_recs = self.get_cached_recommendations(user_id)
        if cached_recs:
            return cached_recs

        movies = self.movie_repo.get_all_movies()
        if not movies:
            raise HTTPException(status_code=404, detail="No movies found")

        num_ratings = self.rating_repo.count_by_user_id(user_id=user_id)

        w_mlp, w_ncf = get_model_weights(num_ratings)

        recommendations = []

        for movie in movies:
            mlp_score = predict_mlp(user, movie)
            ncf_score = predict_ncf(user, movie)

            weighted_score = w_mlp * mlp_score + w_ncf * ncf_score

            recommendations.append({
                "movie_id": movie.id,
                "name": movie.name,
                "score": weighted_score
            })

        recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)[:10]

        self.recommend_repo.save(user_id, recommendations)

        return RecommendationsResponse(
            user_id=user_id,
            recommendations=[RecommendationSchema(**rec) for rec in recommendations]
        )


def get_recommend_service(
        recommend_repo: RecommendRepository = Depends(get_recommend_repository),
        user_repo: UserRepository = Depends(get_user_repository),
        movie_repo: MovieRepository = Depends(get_movie_repository),
        rating_repo: RatingRepository = Depends(get_rating_repository)
):
    return RecommendService(recommend_repo, user_repo, movie_repo, rating_repo)
