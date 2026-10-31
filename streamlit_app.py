import sys
import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

APP_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from recommender import (
    user_cf_recommend,
    item_cf_recommend,
    svd_recommend,
)

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner="Loading models…")
def load_artefacts():
    models_dir = os.path.join(PROJECT_DIR, "models")
    data_dir   = os.path.join(PROJECT_DIR, "data", "processed")

    uim        = joblib.load(os.path.join(models_dir, "user_item_matrix.pkl"))
    u_sim      = joblib.load(os.path.join(models_dir, "user_sim_matrix.pkl"))
    i_sim      = joblib.load(os.path.join(models_dir, "item_sim_matrix.pkl"))
    svd_algo   = joblib.load(os.path.join(models_dir, "svd_model.pkl"))
    train_df   = joblib.load(os.path.join(models_dir, "train_df.pkl"))

    with open(os.path.join(models_dir, "eval_results.json")) as f:
        eval_res = json.load(f)

    cleaned_df = pd.read_csv(os.path.join(data_dir, "cleaned_dataset.csv"))

    return uim, u_sim, i_sim, svd_algo, train_df, eval_res, cleaned_df


uim, u_sim, i_sim, svd_algo, train_df, eval_res, cleaned_df = load_artefacts()

ALL_MOVIES   = eval_res["all_movies"]
GLOBAL_MEAN  = eval_res["global_mean"]
ALL_USER_IDS = sorted(cleaned_df["user_id"].unique().tolist())
TRAIN_USERS  = set(train_df["user_id"].tolist())

st.sidebar.title("🎬 Movie Recommender")

st.sidebar.subheader("Step 1 — Choose a User")
selected_user = st.sidebar.selectbox(
    "User ID",
    options=ALL_USER_IDS,
    help="Select any of the 309 users in the dataset.",
)

st.sidebar.subheader("Step 2 — Choose an Algorithm")
algo_choice = st.sidebar.radio(
    "Recommendation algorithm",
    options=["User-Based CF", "Item-Based CF", "SVD"],
    help=(
        "User-CF: finds users with similar taste.\n"
        "Item-CF: finds movies similar to ones you liked.\n"
        "SVD: matrix factorisation — best for sparse data."
    ),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "ℹ️ **Cold-start note:** Users not seen during training receive "
    "globally popular fallback recommendations (marked ⚠️)."
)

def is_cold_start(user_id):
    return int(user_id) not in TRAIN_USERS


def get_recommendations(user_id, algo, n=5):
    uid = int(user_id)
    if algo == "User-Based CF":
        return user_cf_recommend(
            user_id=uid, n=n,
            user_item_matrix=uim, user_sim_matrix=u_sim,
            train_df=train_df, global_mean=GLOBAL_MEAN,
            all_movies=ALL_MOVIES,
        )
    elif algo == "Item-Based CF":
        return item_cf_recommend(
            user_id=uid, n=n,
            user_item_matrix=uim, item_sim_matrix=i_sim,
            train_df=train_df, global_mean=GLOBAL_MEAN,
            all_movies=ALL_MOVIES,
        )
    else:  # SVD
        return svd_recommend(
            user_id=uid, n=n,
            svd_algo=svd_algo, train_df=train_df,
            all_movies=ALL_MOVIES, global_mean=GLOBAL_MEAN,
        )


st.title("🎬 Personalized Movie Recommendation System")
st.markdown(
    "This application demonstrates three collaborative filtering approaches "
    "trained on a dataset of 309 users and 8 Indian movies. "
    "Select a user and an algorithm in the sidebar to see personalised recommendations."
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["📋 Your Ratings", "🎯 Recommendations", "📊 Model Performance"]
)

with tab1:
    st.subheader(f"Ratings history for User {selected_user}")

    user_ratings = cleaned_df[cleaned_df["user_id"] == selected_user][
        ["movie_title", "rating", "genre", "language", "release_year", "timestamp"]
    ].sort_values("rating", ascending=False).reset_index(drop=True)

    if user_ratings.empty:
        st.info("This user has no ratings in the cleaned dataset.")
    else:
        st.markdown(f"**{len(user_ratings)} rating(s) found.**")

        # Styled dataframe
        def colour_rating(val):
            if val >= 4.0:
                return "background-color: #d4edda; color: #155724;"
            elif val >= 3.0:
                return "background-color: #fff3cd; color: #856404;"
            else:
                return "background-color: #f8d7da; color: #721c24;"

        styled = (
            user_ratings.rename(columns={
                "movie_title": "Movie",
                "rating": "Rating",
                "genre": "Genre",
                "language": "Language",
                "release_year": "Year",
                "timestamp": "Date Rated",
            })
            .style
            .applymap(colour_rating, subset=["Rating"])
            .format({"Rating": "{:.1f}"})
        )
        st.dataframe(styled, use_container_width=True)

    # Cold-start status
    cold = is_cold_start(selected_user)
    if cold:
        st.warning(
            f"⚠️ User {selected_user} was not in the training set. "
            "Recommendations will be globally popular fallbacks, not personalised."
        )
    else:
        n_train = len(train_df[train_df["user_id"] == selected_user])
        st.success(
            f"✅ User {selected_user} has **{n_train}** rating(s) in the training set. "
            "Personalised recommendations are available."
        )

with tab2:
    st.subheader(f"Top recommendations for User {selected_user} — {algo_choice}")

    # Algorithm explanation
    algo_help = {
        "User-Based CF": (
            "**User-Based Collaborative Filtering** finds the 10 most similar users "
            "(by cosine similarity on their rating vectors) and predicts a rating for "
            "each unseen movie as a weighted average of those neighbours' ratings."
        ),
        "Item-Based CF": (
            "**Item-Based Collaborative Filtering** computes cosine similarity between "
            "movies and predicts ratings for unseen movies based on similarity to movies "
            "the user has already rated."
        ),
        "SVD": (
            "**SVD Matrix Factorization** decomposes the user-item rating matrix into "
            "latent factor vectors for users and movies. It handles data sparsity better "
            "than memory-based methods by inferring hidden patterns across all ratings. "
            f"Best params: {eval_res['svd_best_params']}."
        ),
    }
    with st.expander("ℹ️ How does this algorithm work?", expanded=False):
        st.markdown(algo_help[algo_choice])

    with st.spinner("Generating recommendations…"):
        recs = get_recommendations(selected_user, algo_choice, n=5)

    if not recs:
        st.info("No unseen movies available to recommend for this user.")
    else:
        cold = is_cold_start(selected_user)
        if cold:
            st.warning(
                "⚠️ Cold-start user: recommendations below are **fallback suggestions** "
                "based on globally highest-rated movies in the training set, "
                "not personalised predictions."
            )

        # Build recommendation cards
        cols = st.columns(min(len(recs), 5))
        for idx, (col, (movie, pred_rating, is_fallback)) in enumerate(
            zip(cols, recs), start=1
        ):
            with col:
                # Rating bar colour
                if pred_rating >= 4.0:
                    bar_color = "#28a745"
                elif pred_rating >= 3.0:
                    bar_color = "#ffc107"
                else:
                    bar_color = "#dc3545"

                fallback_badge = " ⚠️" if is_fallback else ""
                st.markdown(
                    f"""
                    <div style="border:1px solid #dee2e6; border-radius:8px;
                                padding:14px; text-align:center; height:160px;
                                background:#f8f9fa;">
                        <div style="font-weight:700; font-size:14px;
                                    margin-bottom:8px;">{idx}. {movie}{fallback_badge}</div>
                        <div style="font-size:28px; font-weight:800;
                                    color:{bar_color};">{pred_rating:.2f}</div>
                        <div style="font-size:11px; color:#6c757d;">
                            Predicted rating / 5.0</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Recommendations table
        rec_df = pd.DataFrame(
            [(i + 1, m, f"{r:.2f}", "⚠️ Fallback" if fb else "✅ Personalised")
             for i, (m, r, fb) in enumerate(recs)],
            columns=["Rank", "Movie", "Predicted Rating", "Type"],
        )
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        # User's training history for reference
        seen_movies = list(train_df.loc[train_df["user_id"] == selected_user, "movie_title"])
        if seen_movies:
            st.caption(
                f"Excluded from recommendations (already rated in training): "
                f"{', '.join(sorted(seen_movies))}"
            )

with tab3:
    st.subheader("Model Evaluation Results")

    st.markdown(
        """
        All models were evaluated on the **same held-out test set** (95 rows, 80/20 split,
        `random_state=42`). The test set was never used during training or tuning.

        **RMSE** measures average prediction error (lower is better).
        **Precision@5** measures the fraction of top-5 recommended movies that the user
        actually rated ≥ 4.0 in the test set (higher is better), evaluated on 51 users
        present in both train and test.
        """
    )

    # ---- RMSE bar chart ----
    models_rmse = ["Baseline\n(Global Mean)", "User-Based CF", "Item-Based CF", "SVD"]
    rmse_vals   = [
        eval_res["baseline"]["rmse"],
        eval_res["user_cf"]["rmse"],
        eval_res["item_cf"]["rmse"],
        eval_res["svd"]["rmse"],
    ]
    colours_rmse = ["#6c757d", "#007bff", "#fd7e14", "#28a745"]

    fig_rmse = go.Figure(go.Bar(
        x=models_rmse,
        y=rmse_vals,
        marker_color=colours_rmse,
        text=[f"{v:.4f}" for v in rmse_vals],
        textposition="outside",
    ))
    fig_rmse.update_layout(
        title="RMSE by Model (lower is better)",
        yaxis_title="RMSE",
        yaxis=dict(range=[0, max(rmse_vals) * 1.25]),
        plot_bgcolor="white",
        height=380,
        margin=dict(t=50, b=20),
        showlegend=False,
    )
    fig_rmse.update_xaxes(showgrid=False)
    fig_rmse.update_yaxes(showgrid=True, gridcolor="#e9ecef")

    # ---- Precision@5 bar chart ----
    models_p5 = ["User-Based CF", "Item-Based CF", "SVD"]
    p5_vals   = [
        eval_res["user_cf"]["precision_at_5"],
        eval_res["item_cf"]["precision_at_5"],
        eval_res["svd"]["precision_at_5"],
    ]
    colours_p5 = ["#007bff", "#fd7e14", "#28a745"]

    fig_p5 = go.Figure(go.Bar(
        x=models_p5,
        y=p5_vals,
        marker_color=colours_p5,
        text=[f"{v:.4f}" for v in p5_vals],
        textposition="outside",
    ))
    fig_p5.update_layout(
        title="Precision@5 by Model (higher is better)",
        yaxis_title="Precision@5",
        yaxis=dict(range=[0, max(p5_vals) * 1.5 if max(p5_vals) > 0 else 0.5]),
        plot_bgcolor="white",
        height=380,
        margin=dict(t=50, b=20),
        showlegend=False,
    )
    fig_p5.update_xaxes(showgrid=False)
    fig_p5.update_yaxes(showgrid=True, gridcolor="#e9ecef")

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(fig_rmse, use_container_width=True)
    with col_right:
        st.plotly_chart(fig_p5, use_container_width=True)

    # ---- Results table ----
    st.subheader("Full Results Table")
    notes = eval_res["precision_at_5_notes"]
    results_table = pd.DataFrame([
        {
            "Model": "Global Mean Baseline",
            "RMSE": eval_res["baseline"]["rmse"],
            "Precision@5": "—",
            "Notes": "Predicts training mean (3.6230) for every test row",
        },
        {
            "Model": "User-Based CF",
            "RMSE": eval_res["user_cf"]["rmse"],
            "Precision@5": eval_res["user_cf"]["precision_at_5"],
            "Notes": "Cosine similarity, K=10 neighbours; cold-start → global mean",
        },
        {
            "Model": "Item-Based CF",
            "RMSE": eval_res["item_cf"]["rmse"],
            "Precision@5": eval_res["item_cf"]["precision_at_5"],
            "Notes": "Item cosine similarity; cold-start → global mean",
        },
        {
            "Model": "SVD",
            "RMSE": eval_res["svd"]["rmse"],
            "Precision@5": eval_res["svd"]["precision_at_5"],
            "Notes": f"Best params: {eval_res['svd_best_params']}; cold-start → global mean",
        },
    ])
    st.dataframe(results_table, use_container_width=True, hide_index=True)

    # ---- Disclosures ----
    st.markdown("#### Methodology and Limitations")
    st.markdown(
        f"""
        - **Test set:** 95 rows (20% of 473 cleaned ratings), `random_state=42`.  
          The same 95 rows are used for all three algorithms — no independent resplitting.
        - **Cold-start:** {notes['cold_start_excluded']} of 90 test users were not in the training set.  
          All three models predict the global training mean (3.6230) for these rows.
        - **Precision@5** is evaluated on **{notes['eligible_users']} eligible users**  
          (present in both train and test). Relevance threshold: rating ≥ {notes['relevance_threshold']}.  
          Fixed denominator = {notes['denominator']}; {notes['users_with_lt5_unseen']} users have fewer than 5 unseen movies.
        - **Catalogue size:** With only {notes['total_movies']} movies, recommending top-5 means  
          suggesting 62.5% of the entire catalogue — Precision@5 is inflated vs real-world systems.
        - **Sparsity:** Training matrix is 82.5% sparse, limiting neighbourhood quality in User-CF and Item-CF.  
          SVD handles sparsity better through latent factor decomposition.
        """
    )

