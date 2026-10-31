"""
recommender.py
--------------
Prediction functions for all three collaborative-filtering approaches.
Used by both 02_models.ipynb and the Streamlit application.

All three functions:
  - Exclude movies already present in the user's training history.
  - Return UP TO n recommendations (fewer if the user has fewer unseen movies).
  - For users absent from the training matrix (cold-start), fall back to the
    n globally top-rated movies from training, labelled as fallback predictions.
  - Never use test-set ratings to generate predictions.

Public API
----------
user_cf_recommend(user_id, user_item_matrix, user_sim_matrix,
                  train_df, global_mean, n=5)
item_cf_recommend(user_id, user_item_matrix, item_sim_matrix,
                  train_df, global_mean, n=5)
svd_recommend(user_id, svd_algo, train_df, all_movies,
              global_mean, n=5)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unseen_movies(user_id, train_df, all_movies):
    """Return list of movies the user has NOT rated in training data."""
    seen = set(train_df.loc[train_df['user_id'] == user_id, 'movie_title'])
    return [m for m in all_movies if m not in seen]


def _fallback_recommendations(train_df, all_movies, exclude_movies, n):
    """
    Fallback for cold-start users: return up to n movies from all_movies
    (excluding exclude_movies) ranked by their training mean rating.
    """
    candidates = [m for m in all_movies if m not in exclude_movies]
    movie_means = (
        train_df.groupby('movie_title')['rating']
        .mean()
        .reindex(candidates)
        .sort_values(ascending=False)
    )
    return [(movie, round(float(rating), 4))
            for movie, rating in movie_means.head(n).items()]


# ---------------------------------------------------------------------------
# User-Based Collaborative Filtering
# ---------------------------------------------------------------------------

def user_cf_predict_rating(user_id, movie, user_item_matrix, user_sim_matrix,
                            train_df, global_mean, n_neighbours=10):
    """
    Predict a single rating for (user_id, movie) using user-based CF.
    Returns the global_mean fallback if no neighbour data is available.
    """
    all_users = list(user_item_matrix.index)

    if user_id not in all_users:
        return global_mean, True   # (prediction, is_cold_start)

    user_idx = all_users.index(user_id)
    sim_scores = user_sim_matrix[user_idx]          # similarity to all train users

    # Neighbours: training users who rated this movie
    neighbour_mask = user_item_matrix[movie].notna().values
    neighbour_sims = sim_scores * neighbour_mask    # zero out users who didn't rate movie

    # Sort by descending similarity, take top n_neighbours
    top_idx = np.argsort(neighbour_sims)[::-1][:n_neighbours]
    top_idx = [i for i in top_idx if neighbour_sims[i] > 0]

    if not top_idx:
        # No neighbours rated this movie — fall back to training item mean
        item_mean = train_df.loc[train_df['movie_title'] == movie, 'rating'].mean()
        fallback = item_mean if not np.isnan(item_mean) else global_mean
        return fallback, False

    weights  = neighbour_sims[top_idx]
    ratings  = user_item_matrix[movie].iloc[top_idx].values
    pred     = float(np.dot(weights, ratings) / weights.sum())
    pred     = float(np.clip(pred, 1.0, 5.0))
    return pred, False


def user_cf_recommend(user_id, user_item_matrix, user_sim_matrix,
                      train_df, global_mean, all_movies, n=5):
    """
    Recommend up to n unseen movies for user_id using user-based CF.

    Returns
    -------
    list of (movie_title, predicted_rating, is_fallback)
    """
    unseen = _unseen_movies(user_id, train_df, all_movies)

    # Cold-start: user not in training matrix
    if user_id not in list(user_item_matrix.index):
        seen = set(train_df.loc[train_df['user_id'] == user_id, 'movie_title'])
        recs = _fallback_recommendations(train_df, all_movies, seen, n)
        return [(m, r, True) for m, r in recs]

    predictions = []
    for movie in unseen:
        pred, _ = user_cf_predict_rating(
            user_id, movie, user_item_matrix, user_sim_matrix,
            train_df, global_mean
        )
        predictions.append((movie, round(float(pred), 4)))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [(m, r, False) for m, r in predictions[:n]]


# ---------------------------------------------------------------------------
# Item-Based Collaborative Filtering
# ---------------------------------------------------------------------------

def item_cf_predict_rating(user_id, movie, user_item_matrix, item_sim_matrix,
                            train_df, global_mean):
    """
    Predict a single rating for (user_id, movie) using item-based CF.
    Returns global_mean fallback for cold-start users.
    """
    all_movies_idx = list(user_item_matrix.columns)
    all_users      = list(user_item_matrix.index)

    if user_id not in all_users or movie not in all_movies_idx:
        return global_mean, True

    movie_idx  = all_movies_idx.index(movie)
    item_sims  = item_sim_matrix[movie_idx]         # similarity to all other movies

    # Rated movies by this user in training
    user_row   = user_item_matrix.loc[user_id]
    rated_mask = user_row.notna().values

    weighted_sims    = item_sims * rated_mask
    top_idx          = np.argsort(weighted_sims)[::-1]
    top_idx          = [i for i in top_idx if weighted_sims[i] > 0]

    if not top_idx:
        item_mean = train_df.loc[train_df['movie_title'] == movie, 'rating'].mean()
        fallback  = item_mean if not np.isnan(item_mean) else global_mean
        return fallback, False

    weights = weighted_sims[top_idx]
    ratings = user_row.iloc[top_idx].values
    pred    = float(np.dot(weights, ratings) / weights.sum())
    pred    = float(np.clip(pred, 1.0, 5.0))
    return pred, False


def item_cf_recommend(user_id, user_item_matrix, item_sim_matrix,
                      train_df, global_mean, all_movies, n=5):
    """
    Recommend up to n unseen movies for user_id using item-based CF.

    Returns
    -------
    list of (movie_title, predicted_rating, is_fallback)
    """
    unseen = _unseen_movies(user_id, train_df, all_movies)

    if user_id not in list(user_item_matrix.index):
        seen = set(train_df.loc[train_df['user_id'] == user_id, 'movie_title'])
        recs = _fallback_recommendations(train_df, all_movies, seen, n)
        return [(m, r, True) for m, r in recs]

    predictions = []
    for movie in unseen:
        pred, _ = item_cf_predict_rating(
            user_id, movie, user_item_matrix, item_sim_matrix,
            train_df, global_mean
        )
        predictions.append((movie, round(float(pred), 4)))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [(m, r, False) for m, r in predictions[:n]]


# ---------------------------------------------------------------------------
# SVD (scikit-surprise)
# ---------------------------------------------------------------------------

def _user_in_svd_trainset(user_id, svd_algo):
    """
    Return True if user_id is known to the fitted Surprise SVD trainset.
    The trainset stores raw IDs in whatever type they were fitted with
    (here: int).  Try int first, then str as a fallback for safety.
    """
    for uid in (int(user_id), str(user_id)):
        try:
            svd_algo.trainset.to_inner_uid(uid)
            return True
        except (ValueError, KeyError):
            pass
    return False


def svd_predict_rating(user_id, movie, svd_algo, global_mean):
    """
    Predict a single rating using the fitted Surprise SVD model.
    Explicit cold-start check: if user is NOT in the SVD trainset,
    return global_mean fallback (Surprise may still produce a biased
    estimate for unknown users rather than marking was_impossible).
    """
    if not _user_in_svd_trainset(user_id, svd_algo):
        return global_mean, True
    pred = svd_algo.predict(int(user_id), movie, verbose=False)
    if pred.details.get('was_impossible', False):
        return global_mean, True
    return float(np.clip(pred.est, 1.0, 5.0)), False


def svd_recommend(user_id, svd_algo, train_df, all_movies, global_mean, n=5):
    """
    Recommend up to n unseen movies for user_id using the fitted SVD model.
    Cold-start users (absent from SVD trainset) receive explicit training-mean
    fallback recommendations, not SVD predictions.

    Returns
    -------
    list of (movie_title, predicted_rating, is_fallback)
    """
    unseen = _unseen_movies(user_id, train_df, all_movies)
    if not unseen:
        return []

    # Explicit cold-start detection — do not rely on Surprise's was_impossible alone
    if not _user_in_svd_trainset(user_id, svd_algo):
        seen = set(train_df.loc[train_df['user_id'] == user_id, 'movie_title'])
        recs = _fallback_recommendations(train_df, all_movies, seen, n)
        return [(m, r, True) for m, r in recs]

    predictions = []
    for movie in unseen:
        pred, is_cold = svd_predict_rating(user_id, movie, svd_algo, global_mean)
        # is_cold should be False here (user is in trainset); guard anyway
        predictions.append((movie, round(float(pred), 4), is_cold))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]
