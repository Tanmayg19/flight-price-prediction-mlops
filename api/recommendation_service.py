# ============================================================
# Hotel Recommendation Service
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RECOMMENDATION_MODEL_DIR = (
    BASE_DIR / "models" / "recommendation"
)


# ============================================================
# Artifact Paths
# ============================================================

SVD_MODEL_PATH = (
    RECOMMENDATION_MODEL_DIR / "svd_model.pkl"
)

SVD_SCORES_PATH = (
    RECOMMENDATION_MODEL_DIR / "svd_scores.pkl"
)

TRAIN_INTERACTIONS_PATH = (
    RECOMMENDATION_MODEL_DIR / "train_interactions.pkl"
)

HOTEL_METADATA_PATH = (
    RECOMMENDATION_MODEL_DIR / "hotel_metadata.pkl"
)

POPULARITY_RANKING_PATH = (
    RECOMMENDATION_MODEL_DIR / "popularity_ranking.pkl"
)


# ============================================================
# Load Recommendation Artifacts
# ============================================================

svd_model = joblib.load(SVD_MODEL_PATH)

svd_scores = joblib.load(SVD_SCORES_PATH)

train_interactions = joblib.load(
    TRAIN_INTERACTIONS_PATH
)

hotel_metadata = joblib.load(
    HOTEL_METADATA_PATH
)

popularity_ranking = joblib.load(
    POPULARITY_RANKING_PATH
)


# ============================================================
# Recommendation Function
# ============================================================

def get_hotel_recommendations(
    user_id: int,
    top_n: int = 3
):
    """
    Generate hotel recommendations for a user.

    Known users:
        Personalized recommendations are generated using
        reconstructed SVD preference scores. Hotels already
        present in the user's training history are excluded.

    Unknown users:
        Globally popular hotels are returned as a cold-start
        fallback.

    Parameters
    ----------
    user_id : int
        User code for whom recommendations are requested.

    top_n : int
        Maximum number of recommendations to return.

    Returns
    -------
    pandas.DataFrame
        Ranked hotel recommendations with hotel metadata.
    """

    if top_n < 1:
        raise ValueError(
            "top_n must be greater than or equal to 1."
        )

    # --------------------------------------------------------
    # Known user
    # --------------------------------------------------------

    if user_id in svd_scores.index:

        seen_hotels = set(
            train_interactions.loc[
                train_interactions["userCode"] == user_id,
                "name"
            ]
        )

        candidate_hotels = [
            hotel
            for hotel in svd_scores.columns
            if hotel not in seen_hotels
        ]

        # ----------------------------------------------------
        # Personalized SVD recommendations
        # ----------------------------------------------------

        if candidate_hotels:

            user_scores = (
                svd_scores
                .loc[user_id, candidate_hotels]
                .sort_values(ascending=False)
            )

            ranked_hotels = (
                user_scores
                .head(top_n)
                .rename("recommendation_score")
                .reset_index()
            )

            ranked_hotels.columns = [
                "name",
                "recommendation_score"
            ]

            recommendation_type = (
                "Personalized SVD"
            )

        # ----------------------------------------------------
        # Known user but no unseen hotels
        # ----------------------------------------------------

        else:

            ranked_hotels = pd.DataFrame({
                "name": popularity_ranking[:top_n]
            })

            ranked_hotels[
                "recommendation_score"
            ] = np.nan

            recommendation_type = (
                "Popularity Fallback"
            )

    # --------------------------------------------------------
    # Unknown / cold-start user
    # --------------------------------------------------------

    else:

        ranked_hotels = pd.DataFrame({
            "name": popularity_ranking[:top_n]
        })

        ranked_hotels[
            "recommendation_score"
        ] = np.nan

        recommendation_type = (
            "Popularity Fallback"
        )

    # --------------------------------------------------------
    # Add hotel metadata
    # --------------------------------------------------------

    recommendations = ranked_hotels.merge(
        hotel_metadata,
        on="name",
        how="left"
    )

    recommendations.insert(
        0,
        "userCode",
        user_id
    )

    recommendations.insert(
        1,
        "recommendation_type",
        recommendation_type
    )

    return recommendations