import pandas as pd, sys
sys.path.insert(0, 'src')
from cleaning import clean_dataset

raw = pd.read_csv('data/raw/dataset.csv')
cleaned, removed, log = clean_dataset(raw, verbose=False)

print(f"Raw rows      : {len(raw)}")
print(f"Cleaned rows  : {len(cleaned)}")
n_pairs = removed.groupby(["user_id","movie_title"]).ngroups if not removed.empty else 0
print(f"Removed rows  : {len(removed)} across {n_pairs} distinct pairs")
print()
print("All removed duplicate pairs (sorted by user_id, movie_title):")
if not removed.empty:
    grp = (removed.groupby(["user_id","movie_title"])
                  .agg(removed_count=("rating","count"),
                       removed_ratings=("rating", list),
                       removed_timestamps=("timestamp", list))
                  .reset_index()
                  .sort_values(["user_id","movie_title"]))
    for _, row in grp.iterrows():
        print(f"  user={row['user_id']:>4}  movie={row['movie_title']:<12}  "
              f"removed={row['removed_count']}  "
              f"ratings={row['removed_ratings']}  "
              f"timestamps={row['removed_timestamps']}")
print()
print(f"Unique users  : {cleaned['user_id'].nunique()}")
print(f"Unique movies : {cleaned['movie_title'].nunique()}")
