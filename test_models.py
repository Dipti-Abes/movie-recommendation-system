"""
test_models.py
--------------
Smoke tests for trained models and recommender.py functions.

Tests:
  T1  model files load correctly
  T2  user_cf_recommend returns <=5 unseen movies for warm user
  T3  item_cf_recommend returns <=5 unseen movies for warm user
  T4  svd_recommend returns <=5 unseen movies for warm user
  T5  all three recommend NO movies already in user's training history
  T6  cold-start user receives recommendations without error
  T7  user with 4 unseen movies gets <=4 recommendations (not exactly 5)
  T8  eval_results.json has all required keys and numeric values
"""

import sys, os, json
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from recommender import user_cf_recommend, item_cf_recommend, svd_recommend

PASS = "\u2705 PASS"
FAIL = "\u274c FAIL"

def check(label, cond, detail=""):
    status = PASS if cond else FAIL
    msg = f"  {label}: {status}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    return cond


base = os.path.dirname(os.path.abspath(__file__))

# Load all artefacts
uim      = joblib.load(os.path.join(base, 'models', 'user_item_matrix.pkl'))
u_sim    = joblib.load(os.path.join(base, 'models', 'user_sim_matrix.pkl'))
i_sim    = joblib.load(os.path.join(base, 'models', 'item_sim_matrix.pkl'))
svd_algo = joblib.load(os.path.join(base, 'models', 'svd_model.pkl'))
train_df = joblib.load(os.path.join(base, 'models', 'train_df.pkl'))

with open(os.path.join(base, 'models', 'eval_results.json')) as f:
    eval_res = json.load(f)

GLOBAL_MEAN = eval_res['global_mean']
ALL_MOVIES  = eval_res['all_movies']

# Pick a known warm user (in training matrix)
warm_user = int(uim.index[0])
# Pick a cold user (not in training matrix) from full cleaned data
cleaned = pd.read_csv(os.path.join(base, 'data', 'processed', 'cleaned_dataset.csv'))
from sklearn.model_selection import train_test_split
_, test = train_test_split(cleaned, test_size=0.2, random_state=42)
cold_candidates = set(test['user_id']) - set(train_df['user_id'])
cold_user = int(list(cold_candidates)[0]) if cold_candidates else None

# Find a user with exactly 4 unseen movies
user_with_4_unseen = None
for u in uim.index:
    seen = set(train_df.loc[train_df['user_id'] == u, 'movie_title'])
    if len([m for m in ALL_MOVIES if m not in seen]) == 4:
        user_with_4_unseen = int(u)
        break

print("=" * 60)
print("SMOKE TESTS — Models and Recommender Functions")
print("=" * 60)
print(f"Warm user     : {warm_user}")
print(f"Cold user     : {cold_user}")
print(f"User w/4 unseen: {user_with_4_unseen}")
print()

results = []

# T1: artefact files load
print("T1  Artefact files load correctly")
for name, obj in [('user_item_matrix', uim), ('user_sim_matrix', u_sim),
                  ('item_sim_matrix', i_sim), ('svd_model', svd_algo),
                  ('train_df', train_df)]:
    r = check(f"    {name} loaded", obj is not None)
    results.append(r)

# T2-T4: warm user recommendations
for fn, name, kwargs in [
    (user_cf_recommend, 'User-CF', dict(user_item_matrix=uim, user_sim_matrix=u_sim,
                                        train_df=train_df, global_mean=GLOBAL_MEAN, all_movies=ALL_MOVIES)),
    (item_cf_recommend, 'Item-CF', dict(user_item_matrix=uim, item_sim_matrix=i_sim,
                                        train_df=train_df, global_mean=GLOBAL_MEAN, all_movies=ALL_MOVIES)),
    (svd_recommend,     'SVD',     dict(svd_algo=svd_algo, train_df=train_df,
                                        all_movies=ALL_MOVIES, global_mean=GLOBAL_MEAN)),
]:
    print(f"\nT  {name} warm user (uid={warm_user})")
    recs = fn(user_id=warm_user, n=5, **kwargs)
    r = check(f"    returns list", isinstance(recs, list))
    results.append(r)
    r = check(f"    at most 5 recommendations", len(recs) <= 5, f"got {len(recs)}")
    results.append(r)
    r = check(f"    at least 1 recommendation", len(recs) >= 1, f"got {len(recs)}")
    results.append(r)
    # T5: no seen movies in recommendations
    seen = set(train_df.loc[train_df['user_id'] == warm_user, 'movie_title'])
    rec_movies = [m for m, _, _ in recs]
    overlap = seen & set(rec_movies)
    r = check(f"    no training movies recommended", len(overlap) == 0, f"overlap={overlap}")
    results.append(r)
    # Tuple format
    r = check(f"    each rec is (movie, rating, is_fallback)", all(len(x)==3 for x in recs))
    results.append(r)
    # Ratings in range
    r = check(f"    predicted ratings in [1.0, 5.0]", all(1.0 <= rt <= 5.0 for _, rt, _ in recs))
    results.append(r)

# T6: cold-start user
if cold_user is not None:
    print(f"\nT6  Cold-start user (uid={cold_user})")
    for fn, name, kwargs in [
        (user_cf_recommend, 'User-CF', dict(user_item_matrix=uim, user_sim_matrix=u_sim,
                                            train_df=train_df, global_mean=GLOBAL_MEAN, all_movies=ALL_MOVIES)),
        (item_cf_recommend, 'Item-CF', dict(user_item_matrix=uim, item_sim_matrix=i_sim,
                                            train_df=train_df, global_mean=GLOBAL_MEAN, all_movies=ALL_MOVIES)),
        (svd_recommend,     'SVD',     dict(svd_algo=svd_algo, train_df=train_df,
                                            all_movies=ALL_MOVIES, global_mean=GLOBAL_MEAN)),
    ]:
        recs = fn(user_id=cold_user, n=5, **kwargs)
        r = check(f"    {name} returns without error, >= 1 rec", len(recs) >= 1, f"{len(recs)} recs")
        results.append(r)
        fallbacks = [fb for _, _, fb in recs]
        r = check(f"    {name} marks recs as fallback", all(fallbacks), f"fallbacks={fallbacks}")
        results.append(r)

# T7: user with 4 unseen movies gets <=4 recs
if user_with_4_unseen is not None:
    print(f"\nT7  User with 4 unseen movies (uid={user_with_4_unseen})")
    for fn, name, kwargs in [
        (user_cf_recommend, 'User-CF', dict(user_item_matrix=uim, user_sim_matrix=u_sim,
                                            train_df=train_df, global_mean=GLOBAL_MEAN, all_movies=ALL_MOVIES)),
        (item_cf_recommend, 'Item-CF', dict(user_item_matrix=uim, item_sim_matrix=i_sim,
                                            train_df=train_df, global_mean=GLOBAL_MEAN, all_movies=ALL_MOVIES)),
        (svd_recommend,     'SVD',     dict(svd_algo=svd_algo, train_df=train_df,
                                            all_movies=ALL_MOVIES, global_mean=GLOBAL_MEAN)),
    ]:
        recs = fn(user_id=user_with_4_unseen, n=5, **kwargs)
        r = check(f"    {name} returns <=4 recs (not 5)", len(recs) <= 4, f"got {len(recs)}")
        results.append(r)

# T8: eval_results.json structure
print("\nT8  eval_results.json structure")
for key in ['baseline', 'user_cf', 'item_cf', 'svd']:
    r = check(f"    key '{key}' present", key in eval_res)
    results.append(r)
    rmse = eval_res[key]['rmse']
    r = check(f"    '{key}' rmse is numeric > 0", isinstance(rmse, (int,float)) and rmse > 0, f"rmse={rmse}")
    results.append(r)
p5_keys = ['user_cf', 'item_cf', 'svd']
for key in p5_keys:
    p5 = eval_res[key]['precision_at_5']
    r = check(f"    '{key}' precision_at_5 in [0,1]", isinstance(p5, (int,float)) and 0 <= p5 <= 1, f"p5={p5}")
    results.append(r)

print()
print("=" * 60)
overall = all(results)
print(f"OVERALL: {PASS if overall else FAIL}  ({sum(results)}/{len(results)} checks passed)")
