import json, os

with open('notebooks/01_eda_and_cleaning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                errors.append((i, output.get('ename',''), output.get('evalue','')))

if errors:
    print('ERRORS FOUND:')
    for idx, ename, evalue in errors:
        print(f'  Cell {idx}: {ename}: {evalue}')
else:
    print('No errors in any notebook cell.')

# Check key output files
files = [
    'reports/viz1_rating_distribution.png',
    'reports/viz2_movie_stats.png',
    'reports/viz3_sparsity_heatmap.png',
    'data/processed/cleaned_dataset.csv',
    'data/processed/data_dictionary.csv',
]
print()
for fpath in files:
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    status = "OK" if exists and size > 0 else "MISSING or EMPTY"
    print(f"  {fpath}: {status} ({size} bytes)")
