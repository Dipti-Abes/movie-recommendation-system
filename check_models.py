import json, os

# Check notebook for errors
import sys
sys.path.insert(0, '.')

with open('notebooks/02_models.ipynb', 'r', encoding='utf-8') as f:
    import json as jmod
    nb = jmod.load(f)

errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                errors.append((i, output.get('ename',''), output.get('evalue','')))

if errors:
    print('NOTEBOOK ERRORS:')
    for idx, ename, evalue in errors:
        print(f'  Cell {idx}: {ename}: {evalue}')
else:
    print('Notebook 02_models.ipynb: no errors in any cell.')

# Load and print eval results
with open('models/eval_results.json') as f:
    r = json.load(f)

print()
print('=== EVALUATION RESULTS ===')
print(f"Global mean         : {r['global_mean']:.4f}")
print(f"Baseline RMSE       : {r['baseline']['rmse']}")
print(f"User-CF  RMSE       : {r['user_cf']['rmse']}   P@5: {r['user_cf']['precision_at_5']}")
print(f"Item-CF  RMSE       : {r['item_cf']['rmse']}   P@5: {r['item_cf']['precision_at_5']}")
print(f"SVD      RMSE       : {r['svd']['rmse']}   P@5: {r['svd']['precision_at_5']}")
print(f"SVD best params     : {r['svd_best_params']}")
print()
notes = r['precision_at_5_notes']
print(f"P@5 eligible users  : {notes['eligible_users']}")
print(f"Cold-start excluded : {notes['cold_start_excluded']}")
print(f"Relevance threshold : {notes['relevance_threshold']}")
print(f"Users with <5 unseen: {notes['users_with_lt5_unseen']}")

# Verify all model files exist
print()
files = [
    'models/svd_model.pkl',
    'models/user_item_matrix.pkl',
    'models/user_sim_matrix.pkl',
    'models/item_sim_matrix.pkl',
    'models/train_df.pkl',
    'models/eval_results.json',
]
for fp in files:
    sz = os.path.getsize(fp) if os.path.exists(fp) else 0
    status = 'OK' if sz > 0 else 'MISSING'
    print(f"  {fp}: {status} ({sz} bytes)")
