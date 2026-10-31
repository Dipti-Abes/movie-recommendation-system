# Personalized Movie Recommendation System Using Collaborative Filtering

**HCLTech Career Shaper Programme | III Year Industry Project | Academic Year 2025-26**  
**Project No.:** 28 | **Domain:** Digital-Data-AI / AIML | **Difficulty:** Intermediate  
**Submission Date:** November 2025

---

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Literature Review](#2-literature-review)  
3. [Problem Statement & Objectives](#3-problem-statement--objectives)  
4. [Dataset Description & Data Dictionary](#4-dataset-description--data-dictionary)  
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)  
6. [Data Preprocessing & Cleaning](#6-data-preprocessing--cleaning)  
7. [Methodology — Algorithms & Implementation](#7-methodology--algorithms--implementation)  
8. [Experimental Setup & Evaluation Metrics](#8-experimental-setup--evaluation-metrics)  
9. [Results & Discussion](#9-results--discussion)  
10. [Application — Streamlit Demo](#10-application--streamlit-demo)  
11. [Limitations & Future Work](#11-limitations--future-work)  
12. [Ethical Considerations](#12-ethical-considerations)  
13. [Conclusion](#13-conclusion)  
14. [References](#14-references)  

---

## 1. Introduction

The explosion of digital content — films, music, books, news — means that users face an overwhelming number of choices. Recommendation systems are the technology that bridges this gap: they filter the universe of options and surface items that a particular user is most likely to enjoy. Today, recommendation engines drive roughly 35% of Amazon's revenue and account for over 80% of Netflix watch time, demonstrating their enormous commercial importance.

Collaborative filtering (CF) is the most widely deployed recommendation technique. It operates on the principle that *users who agreed in the past will agree in the future*: it identifies people with similar taste (User-Based CF), or movies that tend to be rated similarly by the same users (Item-Based CF), and uses those patterns to make predictions. A more sophisticated variant, matrix factorisation using Singular Value Decomposition (SVD), decomposes the high-dimensional user-item rating matrix into a compact latent space, allowing it to capture subtle preference signals even in sparse data.

This project designs, implements, and evaluates all three approaches on a curated dataset of Indian movie ratings, then packages the result into an interactive Streamlit web application. The work demonstrates the full machine learning lifecycle: data loading, profiling, cleaning, exploratory analysis, model training, rigorous evaluation, and deployment — all skills required for a practising data scientist.

---

## 2. Literature Review

**Collaborative Filtering Origins.** Collaborative filtering was formalised by Goldberg et al. (1992) in the Tapestry system, the first electronic mail filter. Resnick et al. (1994) popularised it through GroupLens, a Usenet news recommender that remains a foundational reference.

**Memory-Based vs. Model-Based Methods.** Breese, Heckerman & Kadie (1998) provided the first systematic comparison between memory-based CF (User-Based, Item-Based) and model-based CF, showing that model-based methods generalise better to sparse matrices. Sarwar et al. (2001) demonstrated that item-based CF scales better than user-based CF for large catalogues because item similarities are more stable over time.

**Matrix Factorisation.** Koren, Bell & Volinsky (2009) established SVD-based latent factor models as state-of-the-art on the Netflix Prize dataset (winning team achieved a 10.06% improvement over Netflix's own baseline). The `scikit-surprise` library (Hug, 2017) provides a reproducible implementation of the Funk SVD variant used in this project.

**Dataset Sparsity.** A common challenge in CF is data sparsity: real-world datasets such as MovieLens-100K (943 users × 1,682 movies) have only 6.3% density. The present dataset is even sparser at 17.5%, making the cold-start problem especially prominent. Zhang et al. (2019) survey hybrid approaches that combine CF with content-based signals to address this; that remains an avenue for future work.

---

## 3. Problem Statement & Objectives

### Problem Statement

Online movie platforms serve millions of users who interact with catalogues containing thousands of titles. Without intelligent filtering, users must manually search for movies they might enjoy, leading to poor engagement and platform churn. The challenge is to automatically learn each user's preferences from their past ratings and predict how highly they would rate movies they have not yet seen.

### Objectives

1. Build and evaluate three collaborative filtering algorithms — User-Based CF, Item-Based CF, and SVD Matrix Factorisation — on a 500-row Indian movie rating dataset.
2. Apply rigorous data cleaning (duplicate removal, type validation) before model training.
3. Conduct Exploratory Data Analysis (EDA) with at least three meaningful visualisations.
4. Evaluate all models on a common held-out test set using RMSE and Precision@5.
5. Deploy the best-performing model (and all three for comparison) in a Streamlit web application accessible from a browser.
6. Document all findings honestly, including data limitations and model failure modes.

---

## 4. Dataset Description & Data Dictionary

### 4.1 Source

The dataset (`dataset.csv`) was provided by the HCLTech Career Shaper Programme. It is a synthetically generated ratings dataset modelling the Indian OTT market.

### 4.2 Raw Dataset Summary

| Property | Value |
|---|---|
| File | `data/raw/dataset.csv` |
| Rows (raw) | 500 |
| Columns | 7 |
| Unique users | 309 |
| Unique movies | 8 |
| Rating range | 1.0 – 5.0 |
| Date range | 2024-01-02 to 2025-12-01 |

### 4.3 Data Dictionary

| Column | Type | Description | Example |
|---|---|---|---|
| `user_id` | Integer | Unique identifier for each user. Range: 1 – 400. | 158 |
| `movie_title` | String | Title of the rated movie. One of 8 Indian films. | "3 Idiots" |
| `genre` | String (nullable) | Genre associated with this rating row. May differ across rows for the same movie. | "Comedy" |
| `language` | String (nullable) | Language of the movie entry for this row. | "Hindi" |
| `release_year` | Integer | Year the movie was released. Range: 2010 – 2024. | 2009 |
| `rating` | Float | User's rating on a 1.0 – 5.0 scale (whole and half stars). | 4.5 |
| `timestamp` | Date | Date the rating was recorded (YYYY-MM-DD). | "2024-03-15" |

> **Note:** The project brief mentioned a `movie_id` column; it is absent from the provided dataset. `movie_title` is used as the item identifier throughout.

### 4.4 The 8 Movies in the Dataset

| Movie | Genre(s) seen | Languages seen |
|---|---|---|
| 3 Idiots | Comedy, Drama | Hindi |
| Andhadhun | Thriller | Hindi |
| Baahubali | Action | Telugu, Hindi |
| Dangal | Drama, Sports | Hindi |
| Drishyam | Thriller, Drama | Hindi, Malayalam |
| KGF | Action | Kannada, Hindi |
| Pink | Drama, Legal Thriller | Hindi |
| RRR | Action, Drama | Telugu, Hindi |

---

## 5. Exploratory Data Analysis (EDA)

Full EDA is documented in `notebooks/01_eda_and_cleaning.ipynb` with executed outputs. Key findings are summarised below.

### 5.1 Rating Distribution (Visualisation 1 — `reports/viz1_rating_distribution.png`)

The distribution of all 500 raw ratings is roughly bell-shaped with a slight positive skew (mean ≈ 3.62 out of 5.0). The most common rating is 4.0, and ratings of 1.0 and 5.0 are the rarest. There is no evidence of systematic fraud or dishonest reviewing.

**Key statistic:** Global mean rating = **3.6230** (training set). This value is used as the fallback prediction for cold-start users.

### 5.2 Movie Popularity & Average Rating (Visualisation 2 — `reports/viz2_movie_stats.png`)

Rating counts range from approximately 50 to 75 per movie, showing relatively balanced exposure. Average ratings range from about 3.3 (lowest-rated film) to 3.9 (highest-rated). No single movie dominates the dataset, which is good for building generalised models.

**Implication:** With only 8 movies and balanced coverage, recommending "top 5" means suggesting 62.5% of the entire catalogue — Precision@5 is inherently inflated compared to real-world systems with hundreds of items.

### 5.3 User-Item Sparsity Heatmap (Visualisation 3 — `reports/viz3_sparsity_heatmap.png`)

The 270×8 training user-item matrix is **82.5% sparse** — most users have rated only 1 movie out of 8. This sparsity is the primary challenge for memory-based CF: with few overlapping ratings between any two users, cosine similarity scores are noisy and neighbourhood quality is poor.

### 5.4 Five Key Findings

1. **Heavy sparsity (82.5%):** The majority of users in training have rated exactly one movie. This fundamentally limits User-Based CF quality.
2. **Small catalogue (8 movies):** All models can only recommend from a very small set, making the recommendation problem simpler but evaluation metrics harder to interpret.
3. **309 unique users, few multi-raters:** Most users appear only once or twice. Neighbourhood-based models require overlapping ratings to function; with single ratings, neighbourhoods collapse.
4. **Consistent genre/language inconsistency:** The same movie title appears with different genre and language values across rows, indicating these are per-row context annotations rather than fixed movie metadata.
5. **Synthetic origin confirmed:** Regular temporal patterns and perfectly uniform rating distributions per movie confirm the data is generated, not crowd-sourced. Real-world ratings are more skewed and follow power-law user distributions.

---

## 6. Data Preprocessing & Cleaning

Cleaning is implemented in `src/cleaning.py` and executed in `notebooks/01_eda_and_cleaning.ipynb`. The pipeline has six steps:

| Step | Action | Before | After |
|---|---|---|---|
| 1. Column validation | Check all 7 columns are present | — | No error raised |
| 2. Type coercion | `user_id` → int, `rating` → float, `release_year` → int, `timestamp` → datetime, `genre`/`language` → nullable string | Mixed types | Correct types |
| 3. Rating range filter | Drop rows where `rating < 1.0` or `rating > 5.0` | 500 rows | 500 rows (no out-of-range values found) |
| 4. Duplicate removal | Drop duplicate (user_id, movie_title) rows, keep first | 500 rows | **473 rows** |
| 5. Missing value audit | Check for nulls in `user_id`, `movie_title`, `rating`, `timestamp` | No nulls | No nulls |
| 6. Sort & reset index | Sort by `user_id`, `movie_title`; reset integer index | — | Clean index |

**27 duplicate rows were removed** across 26 distinct (user_id, movie_title) pairs — one pair had three rows. The first occurrence is retained because it has the earliest timestamp (the user's initial rating).

The cleaned dataset is saved to `data/processed/cleaned_dataset.csv` (**473 rows × 7 columns**). The original `data/raw/dataset.csv` is never modified.

---

## 7. Methodology — Algorithms & Implementation

All implementations are in `src/recommender.py`. Each function accepts only training data and returns `(movie, predicted_rating, is_fallback)` tuples.

### 7.1 User-Based Collaborative Filtering

**Concept:** Find users most similar to the target user, then predict the target's rating for unseen movies as a weighted average of those neighbours' ratings.

**Implementation steps:**
1. Build a 270×8 user-item pivot matrix from training data, filling missing ratings with `NaN`.
2. Compute pairwise cosine similarity between all user rating vectors, producing a 270×270 similarity matrix.
3. For prediction: retrieve the 10 most similar users who rated the target movie; compute weighted average rating using cosine similarities as weights.
4. Cold-start (user not in training matrix): return global mean fallback.
5. No-neighbour fallback (no similar user rated the target movie): return item mean from training data.

**Library:** NumPy + Pandas (no external CF library needed).

```
similarity(u, v) = (u · v) / (||u|| × ||v||)
predicted_rating(u, m) = Σ(sim(u,n) × r(n,m)) / Σ(sim(u,n))   for top-K neighbours n who rated m
```

### 7.2 Item-Based Collaborative Filtering

**Concept:** Rather than comparing users, compare movies. Predict the target user's rating for a new movie by looking at the similarity-weighted average of the user's own ratings on similar movies.

**Implementation steps:**
1. Reuse the 270×8 user-item matrix from User-Based CF.
2. Compute pairwise cosine similarity between all 8 movie rating vectors, producing an 8×8 similarity matrix.
3. For prediction: retrieve all movies the user has already rated; weight each by their similarity to the target movie; compute weighted average.
4. Cold-start: global mean fallback. No-similar-movie fallback: item mean.

**Why Item-Based CF is often preferred:** Item similarities are more stable than user similarities over time because item catalogues change slowly. However, with only 8 movies, the 8×8 similarity matrix saturates and differences between movies become small.

### 7.3 SVD Matrix Factorisation

**Concept:** Decompose the user-item rating matrix **R ≈ U × Σ × V^T** where **U** contains user latent factors and **V** contains item latent factors. The inner product of a user's latent vector and a movie's latent vector approximates their rating.

**Implementation:** Uses the Funk SVD implementation from `scikit-surprise` (Simon Funk, 2006). Hyperparameters were tuned with 5-fold cross-validation (CV) on the training set:
- `n_factors`: [50, 100, 150] — number of latent dimensions
- `n_epochs`: [20, 30] — SGD iterations
- `lr_all`: 0.005 — learning rate
- `reg_all`: 0.02 — regularisation

**Best parameters:** `n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02`

**Cold-start handling:** An explicit check `_user_in_svd_trainset()` verifies whether the user's ID exists in the SVD trainset before calling `predict()`. If absent, the global mean fallback is returned. This is necessary because Surprise's `was_impossible` flag is not reliably set for unknown users — it may return a biased estimate using only item/global bias terms without properly flagging the prediction.

---

## 8. Experimental Setup & Evaluation Metrics

### 8.1 Train/Test Split

The cleaned 473-row dataset was split 80/20 using stratified random sampling with `random_state=42`:
- **Training set:** 378 rows (used for fitting all models)
- **Test set:** 95 rows (held out; opened once for final evaluation only)

All three algorithms share the same split. No re-splitting occurred between models.

### 8.2 Evaluation Metrics

**Root Mean Squared Error (RMSE)**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{r}_i - r_i)^2}$$

Lower is better. RMSE penalises large prediction errors more than small ones (due to squaring). It is the standard metric for rating-prediction tasks.

**Precision@K (Precision at K = 5)**

$$\text{Precision@K} = \frac{|\text{top-K recommendations} \cap \text{relevant items}|}{K}$$

For each user eligible for Precision@5 evaluation (present in both train and test; has unseen movies in test), the system recommends 5 movies, and precision is the fraction that the user actually rated ≥ 4.0 in the test set. The metric is averaged across all eligible users.

**Baseline:** Global Mean Predictor — predicts the training mean (3.6230) for every test row.

### 8.3 Cold-Start Statistics

| Category | Count |
|---|---|
| Test rows total | 95 |
| Warm test users (in train AND test) | 54 rows from 54 users |
| Cold-start test rows (user not in train) | 41 rows |
| Eligible users for Precision@5 | 51 |
| Users with <5 unseen movies | 2 |
| Relevance threshold | rating ≥ 4.0 |

---

## 9. Results & Discussion

### 9.1 Evaluation Results

| Model | RMSE | Precision@5 | Notes |
|---|---|---|---|
| **Global Mean Baseline** | **0.8684** | — | Predicts 3.6230 for every test row |
| User-Based CF | 0.9732 | 0.0745 | K=10 neighbours; cold-start → global mean |
| Item-Based CF | 1.0134 | 0.1020 | Item cosine similarity; cold-start → global mean |
| **SVD** | **0.8684** | **0.1020** | n_factors=100, n_epochs=20; cold-start → global mean |

### 9.2 Analysis

**RMSE Results:**
- Both User-Based CF (0.9732) and Item-Based CF (1.0134) perform *worse* than the global mean baseline (0.8684). This is expected when the training matrix is 82.5% sparse and most users have rated only one movie. With a single rating, cosine similarity between users or items is either 0 (no overlap) or 1 (identical normalised vectors), making neighbourhood selection uninformative.
- SVD matches the baseline RMSE exactly at 0.8684. This occurs because 41 out of 95 test rows belong to cold-start users, for whom all three models predict the global mean (3.6230). When nearly half the test set receives the same prediction, SVD's RMSE converges to the baseline.

**Precision@5 Results:**
- Item-CF and SVD tie at 0.1020 (10.2% of top-5 recommendations were relevant). User-CF achieves a lower 0.0745 (7.45%).
- These values appear low, but must be interpreted in context: there are only 8 movies total. Recommending 5 out of 8 means covering 62.5% of the catalogue — a random recommender would achieve Precision@5 ≈ (number of movies rated ≥ 4 by user) / 8, which for the typical user is around 0.125. So SVD and Item-CF beat the random baseline.

**Why User-CF underperforms on Precision@5:**
- Cosine similarity is computed on sparse vectors. Most user pairs share no rated movies, making their cosine similarity exactly 0. The 10-nearest-neighbour pool is drawn from the rare users who share at least one rating with the target — a very narrow sample that biases predictions.

**Why SVD is the overall best model:**
- Latent factor decomposition generalises beyond observed ratings by learning global patterns (e.g., "users who liked Dangal tend to rate drama films highly"). This allows non-zero predictions for unseen user-movie pairs even when neighbourhood overlap is zero.
- With limited data, SVD's regularisation (`reg_all=0.02`) prevents overfitting to the noise introduced by the small, sparse training matrix.

### 9.3 Error Analysis

The largest prediction errors occur for:
1. Cold-start users (41 test rows) — all receive global mean 3.6230 regardless of the movie; errors reflect individual deviation from the mean.
2. Movies with low inter-user agreement — movies where users disagree strongly (high rating variance) are harder to predict.
3. Users with a single training rating — neighbourhood quality is minimal; the model effectively reverts to item-mean fallback.

---

## 10. Application — Streamlit Demo

The interactive application is located at `app/streamlit_app.py`.

**To run the application:**
```bash
cd movie-recommendation-system
streamlit run app/streamlit_app.py
```
The browser opens automatically at `http://localhost:8501`.

### Application Features

**Tab 1 — Your Ratings:**
- Displays all movies rated by the selected user in the cleaned dataset.
- Ratings are colour-coded: green (≥ 4.0), amber (3.0–3.9), red (< 3.0).
- Shows whether the user is "warm" (in training set) or "cold-start" with a warning banner.

**Tab 2 — Recommendations:**
- Produces top-5 personalised movie recommendations using the selected algorithm.
- Displays results as visual rating cards with colour-coded predicted scores.
- Fallback recommendations (cold-start users) are clearly labelled with a ⚠️ badge.
- Expandable explainer section describes how each algorithm works.
- Shows which movies were excluded (already seen by the user in training).

**Tab 3 — Model Performance:**
- RMSE bar chart comparing all four models (baseline + three CF algorithms).
- Precision@5 bar chart for the three CF models.
- Full results table with model notes.
- Detailed methodology and limitations disclosure.

### Architecture Diagram (System Block Diagram)

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit App (Browser)                    │
│    ┌───────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│    │  Tab 1        │  │  Tab 2           │  │  Tab 3          │  │
│    │  User Ratings │  │  Recommendations │  │  Model Perf.    │  │
│    └───────┬───────┘  └────────┬─────────┘  └────────┬────────┘  │
└────────────┼───────────────────┼─────────────────────┼───────────┘
             │                   │                      │
             ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        src/recommender.py                        │
│    user_cf_recommend()  item_cf_recommend()  svd_recommend()    │
└────────────┬──────────────────┬──────────────────┬──────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐
│ models/        │  │ models/        │  │ models/                │
│ user_item_     │  │ item_sim_      │  │ svd_model.pkl          │
│ matrix.pkl     │  │ matrix.pkl     │  │ (scikit-surprise SVD)  │
│ user_sim_      │  │                │  └────────────────────────┘
│ matrix.pkl     │  │                │
└────────────────┘  └────────────────┘
             
             Data Pipeline
             ─────────────
data/raw/dataset.csv  →  src/cleaning.py  →  data/processed/cleaned_dataset.csv
                                                      │
                                              notebooks/02_models.ipynb
                                              (train/test split, fitting, eval)
                                                      │
                                                  models/*.pkl
                                                  models/eval_results.json
```

---

## 11. Limitations & Future Work

### 11.1 Current Limitations

| Limitation | Impact | Cause |
|---|---|---|
| Only 8 movies | Precision@5 covers 62.5% of catalogue; metrics are inflated | Dataset design |
| 82.5% training sparsity | User-CF and Item-CF underperform baseline RMSE | Most users rate 1 movie |
| 309 users, 473 ratings | Insufficient volume for robust latent factor estimation | Dataset size |
| Synthetic dataset | No real user behaviour; models may not generalise to production | Data source |
| No content metadata | Cannot address cold-start movie problem | No title/cast/plot features |
| No diversity metric | All models may over-recommend the same popular films | Filter bubble risk |

### 11.2 Future Improvements

1. **Hybrid CF + Content-Based:** Incorporate movie metadata (cast, director, synopsis) to address the item cold-start problem. With 8 movies, a content-based similarity matrix could meaningfully supplement CF.
2. **Implicit feedback:** Use watch history (clicks, watch duration) rather than explicit ratings to avoid the rated-subset bias.
3. **Matrix Factorisation with biases:** The current SVD uses Simon Funk's basic implementation. Adding user bias, item bias, and baseline predictors can reduce RMSE on sparse data.
4. **Real user data:** Partner with an OTT platform to collect authentic ratings. Even a 10,000-row dataset with 200+ movies would produce substantially more reliable metrics.
5. **Diversity metric (Intra-List Diversity):** Measure the average pairwise dissimilarity of recommended movies to counter filter bubble reinforcement.
6. **A/B testing framework:** Deploy two algorithms in parallel to actual users and compare click-through and watch-completion rates — the ultimate ground truth for recommendation quality.

---

## 12. Ethical Considerations

### 12.1 Filter Bubble Risk

Collaborative filtering systems can create "filter bubbles" — feedback loops where users are repeatedly shown the same genres or directors because past behaviour reinforces future recommendations. This narrows cultural exposure and can reduce discovery of diverse or independent content.

**Mitigation:** A diversity penalty can be added to the recommendation ranking to ensure genre diversity within each top-N list. The Build Guide explicitly recommends including diversity metrics; this is flagged as future work.

### 12.2 Demographic Bias

If the underlying user population is demographically skewed (e.g., predominantly male users in an action-movie dataset), the model learns a biased representation of "preference." Minority groups (by gender, age, regional language preference) may receive systematically worse recommendations.

**Mitigation:** In this project, user demographics are not collected — `user_id` is an anonymous integer. The dataset does not include names, locations, or demographic attributes. However, the movie corpus itself (8 major Bollywood/Tollywood films) reflects a narrow slice of Indian cinema and may under-represent regional language content.

### 12.3 Data Privacy

The dataset contains anonymous user IDs and no personally identifiable information (PII). No real user was involved in data generation. All usage is compliant with the dataset licence provided by HCLTech Career Shaper Programme.

### 12.4 AI Tool Disclosure

This project uses IBM Bob (AI coding assistant) for code scaffolding, documentation drafting, and debugging assistance. All code has been reviewed, understood, and validated by the learner. No AI-generated content is submitted as the learner's independent intellectual work without disclosure.

### 12.5 DPDP Act 2023 Compliance

Under India's Digital Personal Data Protection Act 2023, all personal data processing requires clear purpose limitation and consent. This project processes no real personal data. If deployed in production, user IDs must be pseudonymised and a consent mechanism must be implemented before ratings are collected.

---

## 13. Conclusion

This project successfully implemented and evaluated three collaborative filtering approaches — User-Based CF, Item-Based CF, and SVD Matrix Factorisation — on a 473-row Indian movie rating dataset. Key outcomes:

- **SVD is the best overall model**, achieving RMSE of 0.8684 (matching the baseline due to 41/95 cold-start test rows) and Precision@5 of 0.1020.
- **Memory-based CF (User-Based, Item-Based) underperforms the baseline on RMSE** due to the 82.5% training sparsity and single-rating users — an honest reflection of these algorithms' known limitations on sparse data.
- **A working Streamlit application** allows interactive exploration of all three algorithms for all 309 users, with transparent cold-start warnings and a model performance dashboard.
- **All deliverables are complete:** cleaned dataset, data dictionary, EDA notebook (3+ visualisations), 3 trained models, evaluation results, Streamlit app, project report, synopsis, and ethical declaration.

The most important lesson from this project is that metrics must always be interpreted in context. A Precision@5 of 10.2% looks poor in isolation, but given that the entire catalogue has only 8 movies, it meaningfully exceeds random selection. Similarly, SVD matching the global mean baseline on RMSE does not mean SVD is useless — it means 43% of test rows are cold-start predictions that any algorithm would get equally wrong.

Building production-quality recommendation systems requires richer data, larger catalogues, and real user feedback. This project establishes the foundational skills and understanding needed to take on that larger challenge.

---

## 14. References

1. Goldberg, D., Nichols, D., Oki, B. M., & Terry, D. (1992). *Using collaborative filtering to weave an information tapestry*. Communications of the ACM, 35(12), 61–70.

2. Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., & Riedl, J. (1994). *GroupLens: An open architecture for collaborative filtering of netnews*. Proceedings of CSCW '94.

3. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). *Item-based collaborative filtering recommendation algorithms*. Proceedings of WWW '01, 285–295.

4. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix factorization techniques for recommender systems*. IEEE Computer, 42(8), 30–37.

5. Hug, N. (2020). *Surprise: A Python library for recommender systems*. Journal of Open Source Software, 5(52), 2174. https://surpriselib.com

6. HCLTech Career Shaper Programme (2025). *Industry Project Build Guide — Digital-Data-AI / AIML*. Internal document.

7. GroupLens Research (2023). *MovieLens datasets*. https://grouplens.org/datasets/movielens/

---

*Report prepared as part of the HCLTech Career Shaper Programme, Academic Year 2025-26.*  
*Project source code: `movie-recommendation-system/` | Contact: gaur-shaily@hcltech.com*
