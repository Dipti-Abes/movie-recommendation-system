# Personalized Movie Recommendation System Using Collaborative Filtering

**HCLTech Career Shaper Programme · III Year Industry Project · Academic Year 2025-26**  
**Project No.:** 28 | **Domain:** Digital-Data-AI / AIML | **Difficulty:** Intermediate

---

## Project Overview

This project implements and evaluates three collaborative filtering approaches for recommending Indian movies to users:

| Algorithm | RMSE | Precision@5 |
|---|---|---|
| Global Mean Baseline | 0.8684 | — |
| User-Based CF | 0.9732 | 0.0745 |
| Item-Based CF | 1.0134 | 0.1020 |
| **SVD Matrix Factorisation** | **0.8684** | **0.1020** |

> Evaluated on 95 held-out test rows (80/20 split, `random_state=42`). See `reports/project_report.md` Section 9 for full analysis.

---

## Quick Start

### 1. Install dependencies

```bash
cd movie-recommendation-system
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

The browser opens at **http://localhost:8501** automatically.

> **Note:** Models must be trained before running the app. If `models/` is empty, run the notebooks first (see Step 3).

### 3. (Optional) Re-run the full pipeline from scratch

```bash
# Step 1 — EDA and data cleaning
jupyter nbconvert --to notebook --execute notebooks/01_eda_and_cleaning.ipynb

# Step 2 — Train models and evaluate
jupyter nbconvert --to notebook --execute notebooks/02_models.ipynb
```

---

## Project Structure

```
movie-recommendation-system/
├── data/
│   ├── raw/
│   │   ├── dataset.csv                    # Original 500-row dataset (never modified)
│   │   └── data_dictionary_original.csv   # Original data dictionary
│   └── processed/
│       ├── cleaned_dataset.csv            # 473 rows after cleaning
│       └── data_dictionary.csv            # Completed data dictionary (7 columns)
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb          # EDA (3 visualisations) + cleaning pipeline
│   └── 02_models.ipynb                    # Model training, CV tuning, evaluation
├── src/
│   ├── cleaning.py                        # clean_dataset() — 6-step pipeline
│   └── recommender.py                     # user_cf_recommend(), item_cf_recommend(), svd_recommend()
├── models/
│   ├── user_item_matrix.pkl               # 270×8 training user-item pivot
│   ├── user_sim_matrix.pkl                # 270×270 user cosine similarity
│   ├── item_sim_matrix.pkl                # 8×8 item cosine similarity
│   ├── svd_model.pkl                      # Fitted Surprise SVD (n_factors=100)
│   ├── train_df.pkl                       # Training dataframe (378 rows)
│   └── eval_results.json                  # RMSE + Precision@5 for all models
├── app/
│   └── streamlit_app.py                   # Interactive Streamlit application
├── reports/
│   ├── project_report.md                  # Full project report (14 sections, 8+ pages)
│   ├── viz1_rating_distribution.png       # Rating histogram
│   ├── viz2_movie_stats.png               # Count + avg rating per movie
│   └── viz3_sparsity_heatmap.png          # User-item matrix sparsity heatmap
├── synopsis/
│   └── Project_Synopsis.docx             # Completed HCLTech synopsis template
├── docs/
│   └── ethical_declaration.md            # Ethical compliance declaration
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | HCLTech Career Shaper Programme (synthetic) |
| Raw rows | 500 |
| Cleaned rows | 473 (27 duplicate user-movie pairs removed) |
| Unique users | 309 |
| Unique movies | 8 Indian films |
| Rating scale | 1.0 – 5.0 |
| Columns | user_id, movie_title, genre, language, release_year, rating, timestamp |

The 8 movies: 3 Idiots, Andhadhun, Baahubali, Dangal, Drishyam, KGF, Pink, RRR.

---

## Algorithms

### User-Based Collaborative Filtering
Finds the 10 most similar users (cosine similarity on rating vectors) and predicts ratings as a weighted average of neighbours' ratings. Falls back to global mean for cold-start users.

### Item-Based Collaborative Filtering
Computes cosine similarity between movies and predicts a rating based on similarity-weighted average of the user's own ratings on other movies.

### SVD Matrix Factorisation
Uses `scikit-surprise`'s Funk SVD to decompose the user-item matrix into 100 latent factors. Handles sparsity better than memory-based methods. Best hyperparameters found by 5-fold CV: `n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02`.

---

## Cold-Start Handling

Users not present in the training set (39 out of 90 test users) receive **fallback recommendations**: the top-N movies by training mean rating. These are explicitly labelled ⚠️ in the Streamlit app so users understand they are not personalised predictions.

---

## Deliverables Checklist

| Deliverable | File | Status |
|---|---|---|
| Working prototype | `app/streamlit_app.py` | ✅ |
| Project Report (8+ pages) | `reports/project_report.md` | ✅ |
| EDA Notebook (3+ visualisations) | `notebooks/01_eda_and_cleaning.ipynb` | ✅ |
| Cleaned dataset | `data/processed/cleaned_dataset.csv` | ✅ |
| Data dictionary | `data/processed/data_dictionary.csv` | ✅ |
| Project Synopsis | `synopsis/Project_Synopsis.docx` | ✅ |
| Ethical Declaration | `docs/ethical_declaration.md` | ✅ |
| Original dataset (unchanged) | `data/raw/dataset.csv` | ✅ |

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scikit-surprise>=1.1.3
streamlit>=1.28
plotly>=5.17
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
joblib>=1.3
```

Install all with: `pip install -r requirements.txt`

---

## Ethical Notes

- Dataset is fully synthetic — no real user PII involved.
- Filter bubble risk documented in `docs/ethical_declaration.md` and report Section 12.
- AI tool usage (IBM Bob) disclosed in ethical declaration.
- See `docs/ethical_declaration.md` for full compliance declaration.

---

*HCLTech Career Shaper Programme · gaur-shaily@hcltech.com*
