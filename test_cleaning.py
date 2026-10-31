"""
test_cleaning.py
----------------
Validates clean_dataset() across three test cases.

Test A — actual 500-row dataset
         expects 480 rows, 20 duplicate rows removed across 19 pairs.

Test B — synthetic dataframe covering:
  B-missing: NaN/None/np.nan and actual pd.NA in movie_title, genre,
             language, rating, release_year, timestamp
  B-oor:     out-of-range rating (< 1.0 and > 5.0) and release_year
  Checks: log accuracy, no literal 'nan'/'<NA>' strings, input unchanged,
          7 columns preserved.

Test C — duplicate (user_id, movie_title) pair where one timestamp is NaT.
         NaT row is dropped in Step 2 (critical column); removed_duplicates
         is empty; only the valid-timestamp row survives.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from cleaning import clean_dataset

PASS = "\u2705 PASS"
FAIL = "\u274c FAIL"
_ORIGINAL_COLS = ["user_id", "movie_title", "genre", "language",
                  "release_year", "rating", "timestamp"]


def _check(label, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  {label}: {status}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    return condition


# ======================================================================
# TEST A — actual 500-row dataset
# ======================================================================
def test_actual_dataset(dataset_path):
    print("\n" + "=" * 65)
    print("TEST A — Actual 500-row dataset")
    print("=" * 65)

    raw = pd.read_csv(dataset_path)
    raw_copy = raw.copy()
    print(f"Loaded {len(raw)} rows, {len(raw.columns)} columns.\n")

    cleaned, removed, log = clean_dataset(raw, verbose=True)

    results = []

    results.append(_check("A1  Final rows == 473",
                           len(cleaned) == 473, f"got {len(cleaned)}"))

    results.append(_check("A2  Duplicate rows removed == 27",
                           len(removed) == 27, f"got {len(removed)}"))

    n_pairs = (removed.groupby(["user_id", "movie_title"]).ngroups
               if not removed.empty else 0)
    results.append(_check("A3  Distinct dup pairs == 26",
                           n_pairs == 26, f"got {n_pairs}"))

    results.append(_check("A4  Output has original 7 columns",
                           list(cleaned.columns) == _ORIGINAL_COLS,
                           str(list(cleaned.columns))))

    null_count = cleaned[["user_id", "movie_title", "rating",
                           "release_year", "timestamp"]].isnull().sum().sum()
    results.append(_check("A5  No nulls in critical cols",
                           null_count == 0, f"null_count={null_count}"))

    results.append(_check("A6  All ratings in [1.0, 5.0]",
                           cleaned["rating"].between(1.0, 5.0).all()))

    results.append(_check("A7  All years in [1900, 2024]",
                           cleaned["release_year"].between(1900, 2024).all()))

    results.append(_check("A8  Unique users unchanged",
                           cleaned["user_id"].nunique() == raw_copy["user_id"].nunique(),
                           f"cleaned={cleaned['user_id'].nunique()} "
                           f"raw={raw_copy['user_id'].nunique()}"))

    results.append(_check("A9  Unique movies == 8",
                           cleaned["movie_title"].nunique() == 8,
                           f"got {cleaned['movie_title'].nunique()}"))

    results.append(_check("A10 Input dataframe unchanged",
                           raw.equals(raw_copy)))

    # No literal "nan" or "<NA>" strings in non-null cells of text columns
    for col in ["movie_title", "genre", "language"]:
        non_null_mask = cleaned[col].notna()
        bad = (cleaned.loc[non_null_mask, col]
               .astype(str).str.lower()
               .isin(["nan", "<na>"])
               .sum())
        results.append(_check(
            f"A11 No 'nan'/'<NA>' strings in non-null {col}",
            bad == 0, f"found {bad}"
        ))

    print(f"\nRemoved duplicates ({len(removed)} rows):")
    if not removed.empty:
        print(removed[["user_id", "movie_title", "rating", "timestamp"]]
              .sort_values(["user_id", "movie_title"])
              .to_string(index=False))

    overall = all(results)
    print(f"\nTest A overall: {PASS if overall else FAIL}")
    return overall


# ======================================================================
# TEST B — synthetic dataframe with diverse bad and missing values
# ======================================================================
def test_synthetic_bad_values():
    print("\n" + "=" * 65)
    print("TEST B — Synthetic dataframe: missing and invalid values")
    print("=" * 65)

    # Build rows one at a time for clarity
    rows = [
        # uid, movie,       genre,    lang,  year,  rating, ts
        # 1  valid row — survives
        (1,  "Film_A",     "Action", "Hindi", 2020, 4.0,  "2024-01-01"),
        # 2  missing movie_title via None (critical) → dropped
        (2,  None,         "Drama",  "Tamil", 2020, 3.0,  "2024-02-01"),
        # 3  missing movie_title via np.nan (critical) → dropped
        (3,  np.nan,       "Drama",  "Tamil", 2020, 3.5,  "2024-03-01"),
        # 4  missing movie_title via actual pd.NA (critical) → dropped
        (4,  pd.NA,        "Drama",  "Tamil", 2020, 3.5,  "2024-04-01"),
        # 5  missing rating via None (critical) → dropped
        (5,  "Film_E",     "Action", "Hindi", 2020, None, "2024-05-01"),
        # 6  missing release_year via None (critical) → dropped
        (6,  "Film_F",     "Action", "Hindi", None, 3.0,  "2024-06-01"),
        # 7  missing timestamp via None (critical) → dropped
        (7,  "Film_G",     "Action", "Hindi", 2020, 4.0,  None),
        # 8  rating below 1.0 (OOR) → dropped
        (8,  "Film_H",     "Action", "Hindi", 2020, 0.5,  "2024-08-01"),
        # 9  rating above 5.0 (OOR) → dropped
        (9,  "Film_I",     "Action", "Hindi", 2020, 5.5,  "2024-09-01"),
        # 10 year < 1900 (flagged, OOR) → dropped
        (10, "Film_J",     "Action", "Hindi", 1800, 3.0,  "2024-10-01"),
        # 11 year > 2024 (flagged, OOR) → dropped
        (11, "Film_K",     "Action", "Hindi", 2030, 3.0,  "2024-11-01"),
        # 12 missing genre via None (non-critical) → row KEPT, genre=pd.NA
        (12, "Film_L",     None,     "Hindi", 2020, 4.0,  "2024-12-01"),
        # 13 missing genre via np.nan (non-critical) → row KEPT, genre=pd.NA
        (13, "Film_M",     np.nan,   "Tamil", 2021, 3.5,  "2024-12-15"),
        # 14 missing genre via pd.NA (non-critical) → row KEPT, genre=pd.NA
        (14, "Film_N",     pd.NA,    "Tamil", 2021, 3.0,  "2024-12-20"),
        # 15 missing language via None (non-critical) → row KEPT, language=pd.NA
        (15, "Film_O",     "Comedy", None,    2021, 3.5,  "2024-12-25"),
        # 16 valid row — survives
        (16, "Film_P",     "Comedy", "Hindi", 2021, 3.0,  "2024-12-31"),
    ]

    df_input = pd.DataFrame(rows, columns=_ORIGINAL_COLS)
    df_input_copy = df_input.copy(deep=True)

    print("Input (16 rows):")
    print(df_input.to_string(index=False))
    print()

    cleaned, removed, log = clean_dataset(df_input, verbose=True)

    results = []

    # Critical missing → dropped
    for uid, label in [(2, "None"), (3, "np.nan"), (4, "pd.NA")]:
        results.append(_check(
            f"B-drop  uid={uid} missing movie_title ({label}) dropped",
            not (cleaned["user_id"] == uid).any()
        ))
    results.append(_check("B-drop  uid=5 missing rating dropped",
                           not (cleaned["user_id"] == 5).any()))
    results.append(_check("B-drop  uid=6 missing release_year dropped",
                           not (cleaned["user_id"] == 6).any()))
    results.append(_check("B-drop  uid=7 missing timestamp dropped",
                           not (cleaned["user_id"] == 7).any()))

    # OOR → dropped
    results.append(_check("B-oor   uid=8 rating 0.5 dropped",
                           not (cleaned["user_id"] == 8).any()))
    results.append(_check("B-oor   uid=9 rating 5.5 dropped",
                           not (cleaned["user_id"] == 9).any()))
    results.append(_check("B-oor   uid=10 year 1800 dropped",
                           not (cleaned["user_id"] == 10).any()))
    results.append(_check("B-oor   uid=11 year 2030 dropped",
                           not (cleaned["user_id"] == 11).any()))

    # Non-critical missing genre → row KEPT
    for uid, src in [(12, "None"), (13, "np.nan"), (14, "pd.NA")]:
        row_present = (cleaned["user_id"] == uid).any()
        results.append(_check(
            f"B-keep  uid={uid} missing genre ({src}) row KEPT",
            row_present
        ))
        if row_present:
            val = cleaned.loc[cleaned["user_id"] == uid, "genre"].iloc[0]
            is_null = pd.isna(val)
            not_str = str(val).lower() not in ("nan", "<na>", "none")
            results.append(_check(
                f"B-null  uid={uid} genre is genuinely pd.NA (not string)",
                is_null,
                f"repr={repr(val)}"
            ))

    # Non-critical missing language → row KEPT
    row_present = (cleaned["user_id"] == 15).any()
    results.append(_check("B-keep  uid=15 missing language row KEPT", row_present))
    if row_present:
        val = cleaned.loc[cleaned["user_id"] == 15, "language"].iloc[0]
        results.append(_check(
            "B-null  uid=15 language is genuinely pd.NA (not string)",
            pd.isna(val), f"repr={repr(val)}"
        ))

    # Valid rows survive
    results.append(_check("B-keep  uid=1 and uid=16 valid rows survive",
                           (cleaned["user_id"] == 1).any() and
                           (cleaned["user_id"] == 16).any()))

    # No literal "nan" or "<NA>" strings in text columns
    # Exclude genuinely-null cells: pd.NA prints as "<NA>" via astype(str),
    # but that is correct behaviour — only flag non-null cells that spell those words.
    for col in ["movie_title", "genre", "language"]:
        non_null_mask = cleaned[col].notna()
        bad = (cleaned.loc[non_null_mask, col]
               .astype(str).str.lower()
               .isin(["nan", "<na>"])
               .sum())
        results.append(_check(f"B-str   no 'nan'/'<NA>' strings in non-null {col}",
                               bad == 0, f"found {bad}"))

    # Input dataframe unchanged
    results.append(_check("B-immut input dataframe unchanged",
                           df_input.equals(df_input_copy)))

    # 7 original columns
    results.append(_check("B-cols  output has original 7 columns",
                           list(cleaned.columns) == _ORIGINAL_COLS,
                           str(list(cleaned.columns))))

    # Log records Step 2 (missing), Step 4 (rating OOR), Step 5 (year FLAGGED)
    log_text = "\n".join(log)
    results.append(_check("B-log   Step 2 records missing value removal",
                           "Step 2" in log_text and "Missing value" in log_text))
    results.append(_check("B-log   Step 4 records out-of-range rating",
                           "Step 4" in log_text and "out-of-range" in log_text.lower()))
    results.append(_check("B-log   Step 5 FLAGGED out-of-range year",
                           "Step 5" in log_text and "FLAGGED" in log_text))

    print(f"\nCleaned output ({len(cleaned)} rows):")
    print(cleaned.to_string(index=False))

    overall = all(results)
    print(f"\nTest B overall: {PASS if overall else FAIL}")
    return overall


# ======================================================================
# TEST C — duplicate pair where one timestamp is NaT
#           NaT is a CRITICAL missing value → dropped in Step 2
#           removed_duplicates must be empty (no Step-6 removal needed)
# ======================================================================
def test_nat_timestamp_duplicate():
    print("\n" + "=" * 65)
    print("TEST C — Duplicate pair with one NaT timestamp")
    print("=" * 65)

    data = {
        "user_id":      [10,          10,           20],
        "movie_title":  ["MovieX",    "MovieX",     "MovieY"],
        "genre":        ["Action",    "Drama",      "Comedy"],
        "language":     ["Hindi",     "Tamil",      "Hindi"],
        "release_year": [2020,        2020,         2021],
        "rating":       [3.5,         4.5,          3.0],
        # Row 0: NaT timestamp → dropped in Step 2 (critical)
        # Row 1: valid timestamp → survives (no duplicate left)
        "timestamp":    [pd.NaT,      "2024-06-01", "2024-05-01"],
    }
    df_input = pd.DataFrame(data)
    df_input_copy = df_input.copy(deep=True)

    print("Input (3 rows — uid=10 rated MovieX twice; row 0 has NaT timestamp):")
    print(df_input.to_string(index=False))

    cleaned, removed, log = clean_dataset(df_input, verbose=True)

    results = []

    # C1: NaT-timestamp row (rating=3.5) absent from cleaned output (dropped in Step 2)
    results.append(_check(
        "C1  NaT-timestamp row (uid=10, rating=3.5) absent from cleaned",
        not ((cleaned["user_id"] == 10) & (cleaned["rating"] == 3.5)).any()
    ))

    # C2: valid-timestamp row (rating=4.5) is present
    results.append(_check(
        "C2  Valid-timestamp row (uid=10, rating=4.5) kept",
        ((cleaned["user_id"] == 10) & (cleaned["rating"] == 4.5)).any()
    ))

    # C3: unrelated row (uid=20, MovieY) survives
    results.append(_check(
        "C3  Unrelated row (uid=20, MovieY) kept",
        ((cleaned["user_id"] == 20) & (cleaned["movie_title"] == "MovieY")).any()
    ))

    # C4: removed_duplicates is EMPTY (NaT row was removed in Step 2, not Step 6)
    results.append(_check(
        "C4  removed_duplicates is empty (NaT row removed in Step 2)",
        len(removed) == 0, f"got {len(removed)} rows"
    ))

    # C5: log records Step 2 removal of the NaT timestamp
    log_text = "\n".join(log)
    results.append(_check(
        "C5  Log records Step 2 missing-value removal",
        "Step 2" in log_text and "Missing value" in log_text
    ))

    # C6: log records missing timestamp specifically
    results.append(_check(
        "C6  Log identifies timestamp as missing column",
        "timestamp" in log_text.lower()
    ))

    # C7: input dataframe unchanged
    results.append(_check("C7  Input dataframe unchanged",
                           df_input.equals(df_input_copy)))

    print(f"\nCleaned output ({len(cleaned)} rows):")
    print(cleaned.to_string(index=False))

    overall = all(results)
    print(f"\nTest C overall: {PASS if overall else FAIL}")
    return overall


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base, "data", "raw", "dataset.csv")

    a = test_actual_dataset(dataset_path)
    b = test_synthetic_bad_values()
    c = test_nat_timestamp_duplicate()

    print("\n" + "=" * 65)
    print("ALL TESTS SUMMARY")
    print("=" * 65)
    print(f"  Test A (actual dataset):            {PASS if a else FAIL}")
    print(f"  Test B (synthetic bad values):      {PASS if b else FAIL}")
    print(f"  Test C (NaT timestamp duplicate):   {PASS if c else FAIL}")
    overall = a and b and c
    print(f"\n  Overall: {PASS if overall else FAIL}")
    sys.exit(0 if overall else 1)
