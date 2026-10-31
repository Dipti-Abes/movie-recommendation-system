import pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from cleaning import clean_dataset

raw = pd.read_csv("data/raw/dataset.csv")
cleaned, removed, log = clean_dataset(raw, verbose=False)

# Save cleaned dataset - genre/language are StringDtype; convert to plain object
# so the CSV contains clean strings (or empty for pd.NA)
out = cleaned.copy()
for col in ["genre", "language"]:
    out[col] = out[col].astype(object)

out.to_csv("data/processed/cleaned_dataset.csv", index=False)
print(f"Saved data/processed/cleaned_dataset.csv  ({len(out)} rows, {len(out.columns)} columns)")

# Quick verification
verify = pd.read_csv("data/processed/cleaned_dataset.csv")
print(f"Re-read verification: {len(verify)} rows, columns = {list(verify.columns)}")
assert len(verify) == 473, f"Expected 473 rows, got {len(verify)}"
assert list(verify.columns) == ["user_id","movie_title","genre","language","release_year","rating","timestamp"]
print("Verification PASSED")
