"""
cleaning.py
-----------
Reusable cleaning function for the Movie Recommendation dataset.
Used by both the EDA notebook and the Streamlit application.

Rules (approved in Plan v4.0):
1. Fix data types (coerce bad values to NaN/NaT).
   Text columns cast with astype("string") (nullable StringDtype) so that
   actual pd.NA values never become the literal string "<NA>" or "nan".
2. Drop rows missing any CRITICAL column (user_id, movie_title, rating,
   release_year, timestamp). Non-critical (genre, language) missing values
   are preserved as pd.NA and the row is kept.
3. Standardise text columns (strip whitespace) — after NaN removal.
4. Validate rating range [1.0, 5.0].
5. Validate release_year range [1900, 2024] — flag first, then remove.
6. Resolve duplicate (user_id, movie_title) pairs with stable mergesort;
   all timestamps are valid by this step (NaT rows removed in Step 2).

Returns: (cleaned_df, removed_duplicates_df, log_lines)
"""

import pandas as pd

_CRITICAL_COLS   = ["user_id", "movie_title", "rating", "release_year", "timestamp"]
_NON_CRITICAL_TEXT = ["genre", "language"]
# Sentinel strings that represent "no value" in any text column
_SENTINELS = {"", "nan", "none", "na", "n/a", "<na>"}


def _to_nullable_string(series: pd.Series) -> pd.Series:
    """
    Convert a Series to nullable StringDtype.
    - True NaN / None / pd.NA  → pd.NA  (stays missing)
    - Sentinel strings          → pd.NA
    - Everything else           → stripped string
    Crucially, astype("string") is used (not astype(str)) so pd.NA
    never becomes the literal string '<NA>'.
    """
    # Cast to nullable string: pd.NA stays pd.NA, real values become strings
    s = series.astype("string")
    # Strip whitespace from non-missing values
    s = s.str.strip()
    # Replace sentinel strings with pd.NA
    s = s.where(~s.str.lower().isin(_SENTINELS), other=pd.NA)
    return s


def clean_dataset(df: pd.DataFrame, verbose: bool = True):
    """
    Clean the raw movie-ratings dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from data/raw/dataset.csv.
    verbose : bool
        If True, print the cleaning log to stdout.

    Returns
    -------
    cleaned : pd.DataFrame
        Cleaned dataframe with exactly the original 7 columns.
        genre and language use StringDtype; missing values are pd.NA.
    removed_duplicates : pd.DataFrame
        Rows dropped because a newer rating existed for the same
        (user_id, movie_title) pair.  NaT-timestamp rows are removed
        in Step 2 and do NOT appear here.
    log : list[str]
        One string per log entry (also printed when verbose=True).
    """
    log = []
    removed_duplicates = pd.DataFrame()

    # Work on a copy — never mutate the caller's dataframe
    df = df.copy()
    initial_rows = len(df)
    log.append(f"Step 0 — Raw input rows: {initial_rows}")

    # ------------------------------------------------------------------
    # Step 1: Fix data types
    # ------------------------------------------------------------------
    df["user_id"]      = pd.to_numeric(df["user_id"],      errors="coerce")
    df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["timestamp"]    = pd.to_datetime(df["timestamp"],   errors="coerce")

    # All three text columns: use nullable StringDtype so pd.NA stays pd.NA
    df["movie_title"] = _to_nullable_string(df["movie_title"])
    df["genre"]       = _to_nullable_string(df["genre"])
    df["language"]    = _to_nullable_string(df["language"])

    log.append(
        "Step 1 — Data types coerced: user_id->numeric, rating->float, "
        "release_year->numeric, timestamp->datetime. "
        "movie_title / genre / language cast with astype('string') so pd.NA "
        "never becomes '<NA>' or 'nan'. Sentinels normalised to pd.NA."
    )

    # ------------------------------------------------------------------
    # Step 2: Drop rows missing any CRITICAL column value
    # ------------------------------------------------------------------
    missing_per_col   = df[_CRITICAL_COLS].isnull().sum()
    rows_with_missing = int(df[_CRITICAL_COLS].isnull().any(axis=1).sum())

    if rows_with_missing > 0:
        missing_detail = {
            col: int(cnt)
            for col, cnt in missing_per_col.items()
            if cnt > 0
        }
        df = df.dropna(subset=_CRITICAL_COLS)
        log.append(
            f"Step 2 — Missing value removal: {rows_with_missing} row(s) dropped "
            f"(critical columns only: {_CRITICAL_COLS}). "
            f"Missing counts per column: {missing_detail}. "
            f"Rows remaining: {len(df)}"
        )
    else:
        log.append(
            f"Step 2 — Missing value check: no missing values in critical columns "
            f"{_CRITICAL_COLS}. Non-critical columns (genre, language) may still "
            f"contain pd.NA — those rows are kept. Rows unchanged: {len(df)}"
        )

    # Log non-critical missing counts (informational; rows are kept)
    nc_missing = {
        col: int(df[col].isna().sum())
        for col in _NON_CRITICAL_TEXT
        if df[col].isna().sum() > 0
    }
    if nc_missing:
        log.append(
            f"Step 2b — Non-critical missing values (rows kept): {nc_missing}"
        )

    # ------------------------------------------------------------------
    # Step 3: Text standardisation — after NaN removal
    # ------------------------------------------------------------------
    # Re-apply strip (already done in Step 1, but safe to repeat after dropna)
    df["movie_title"] = df["movie_title"].astype("string").str.strip()
    for col in _NON_CRITICAL_TEXT:
        df[col] = df[col].astype("string")   # pd.NA preserved

    log.append(
        "Step 3 — Text standardisation complete. "
        "movie_title stripped. genre/language remain StringDtype with pd.NA preserved."
    )

    # ------------------------------------------------------------------
    # Step 4: Validate rating range [1.0, 5.0]
    # ------------------------------------------------------------------
    out_of_range_rating = df[(df["rating"] < 1.0) | (df["rating"] > 5.0)]
    n_oor_rating = len(out_of_range_rating)
    df = df[(df["rating"] >= 1.0) & (df["rating"] <= 5.0)]
    log.append(
        f"Step 4 — Rating range [1.0, 5.0]: "
        f"{n_oor_rating} out-of-range row(s) removed "
        f"(NaN ratings already handled in Step 2). "
        f"Rows remaining: {len(df)}"
    )

    # ------------------------------------------------------------------
    # Step 5: Validate release_year range [1900, 2024] — flag first
    # ------------------------------------------------------------------
    below_range = df[df["release_year"] < 1900]
    above_range = df[df["release_year"] > 2024]
    n_below  = len(below_range)
    n_above  = len(above_range)
    n_flagged = n_below + n_above

    if n_flagged > 0:
        flagged     = pd.concat([below_range, above_range])
        flag_detail = (
            flagged[["user_id", "movie_title", "release_year"]].to_dict("records")
        )
        df = df[(df["release_year"] >= 1900) & (df["release_year"] <= 2024)]
        log.append(
            f"Step 5 — release_year [1900, 2024]: FLAGGED {n_flagged} row(s) "
            f"(below 1900: {n_below}, above 2024: {n_above}). "
            f"Flagged records: {flag_detail}. "
            f"Decision: removed as implausible release years. "
            f"Rows remaining: {len(df)}"
        )
    else:
        log.append(
            f"Step 5 — release_year [1900, 2024]: "
            f"no out-of-range values found. Rows unchanged: {len(df)}"
        )
    n_dropped_year = n_flagged

    # ------------------------------------------------------------------
    # Step 6: Resolve duplicate (user_id, movie_title) pairs
    #         All timestamps are valid by now (NaT rows dropped in Step 2).
    #         Stable mergesort for deterministic tie-breaking.
    # ------------------------------------------------------------------
    before_dedup = len(df)

    df = df.sort_values(
        "timestamp", ascending=True, na_position="first", kind="mergesort"
    )
    df["_is_older_dup"] = df.duplicated(
        subset=["user_id", "movie_title"], keep="last"
    )

    removed_duplicates = (
        df[df["_is_older_dup"]].drop(columns=["_is_older_dup"]).copy()
    )
    df = df[~df["_is_older_dup"]].drop(columns=["_is_older_dup"])

    dropped_dups = before_dedup - len(df)
    n_distinct_pairs = (
        removed_duplicates.groupby(["user_id", "movie_title"]).ngroups
        if not removed_duplicates.empty else 0
    )

    log.append(
        f"Step 6 — Duplicate resolution (stable mergesort, all timestamps valid): "
        f"{dropped_dups} row(s) removed across {n_distinct_pairs} distinct pair(s). "
        f"Decision: kept most recent timestamp. "
        f"Rows remaining: {len(df)}"
    )

    # ------------------------------------------------------------------
    # Restore original column order and reset index
    # ------------------------------------------------------------------
    original_cols = [
        "user_id", "movie_title", "genre", "language",
        "release_year", "rating", "timestamp",
    ]
    df = df[original_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_removed = initial_rows - len(df)
    log.append("")
    log.append("SUMMARY")
    log.append(f"  Input rows              : {initial_rows}")
    log.append(f"  Removed — missing       : {rows_with_missing}")
    log.append(f"  Removed — rating OOR    : {n_oor_rating}")
    log.append(f"  Removed — year OOR      : {n_dropped_year}")
    log.append(f"  Removed — duplicates    : {dropped_dups}")
    log.append(f"  Total removed           : {total_removed}")
    log.append(f"  Final cleaned rows      : {len(df)}")
    log.append(f"  Columns preserved       : {list(df.columns)}")

    if verbose:
        print("\n" + "=" * 65)
        print("CLEANING LOG")
        print("=" * 65)
        for line in log:
            print(line)
        print("=" * 65 + "\n")

    return df, removed_duplicates, log
