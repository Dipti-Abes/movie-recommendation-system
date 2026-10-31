import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/processed/cleaned_dataset.csv')
train, test = train_test_split(df, test_size=0.2, random_state=42)

print(f"Total rows : {len(df)}")
print(f"Train rows : {len(train)}")
print(f"Test rows  : {len(test)}")
print()

train_users = set(train['user_id'])
test_users  = set(test['user_id'])
both_users  = train_users & test_users
cold_users  = test_users - train_users
print(f"Users in train                           : {len(train_users)}")
print(f"Users in test                            : {len(test_users)}")
print(f"Users in BOTH (eligible for Precision@K) : {len(both_users)}")
print(f"Users in test only (cold-start)          : {len(cold_users)}")
print()

warm_test = test[test['user_id'].isin(train_users)]
cold_test  = test[~test['user_id'].isin(train_users)]
print(f"Test rows with warm user (CF prediction) : {len(warm_test)}")
print(f"Test rows with cold user (fallback)      : {len(cold_test)}")
print()

relevant = test[test['user_id'].isin(train_users) & (test['rating'] >= 4.0)]
n_relevant_users = relevant['user_id'].nunique()
print(f"Relevant test rows (warm, rating >= 4.0)   : {len(relevant)}")
print(f"Eligible users with >= 1 relevant rating   : {n_relevant_users}")
print()

print(f"Movies in train : {sorted(train['movie_title'].unique())}")
print(f"Movies in test  : {sorted(test['movie_title'].unique())}")
print()

global_mean = train['rating'].mean()
print(f"Training global mean rating : {global_mean:.4f}")
print()
print("Per-movie mean (training only):")
print(train.groupby('movie_title')['rating'].mean().round(4).to_string())
print()

# Ratings per user breakdown in train vs test
rpu_train = train.groupby('user_id').size()
rpu_test  = test.groupby('user_id').size()
print("Train: ratings-per-user distribution:")
print(rpu_train.value_counts().sort_index().to_string())
print()
print("Test: ratings-per-user distribution:")
print(rpu_test.value_counts().sort_index().to_string())
