#!/bin/zsh
# Rebuild everything downstream of the trained trials: ensemble + figures, extensions, beta network, report.
set -e
cd "$(dirname "$0")"
. .venv/bin/activate
echo "== ensemble + figures";  python make_figures.py 2>&1 | grep -v -E 'WARNING|absl|oneDNN'
echo "== extensions";          python extensions.py results/our_ensemble
echo "== beta network (MPS)";  python beta_ffn.py results/our_ensemble 3 mps
echo "== numbers.tex";         python fill_numbers.py > /dev/null
echo "== report";              (cd report && tectonic report.tex 2>&1 | grep -E -i 'error|undefined' || true; pdfinfo report.pdf | grep Pages)
echo "== published-series check"
python - <<'EOF'
import pandas as pd, numpy as np, json
df = pd.read_excel('data/Data.xlsx', header=6).dropna(subset=['SDF']); df['date'] = pd.to_datetime(df['date']); sdf = df.set_index('date')['SDF']
sp = {'train': sdf[:'1986-12'], 'valid': sdf['1987-01':'1991-12'], 'test': sdf['1992-01':]}
sr = lambda f: float(np.mean(f) / np.std(f)); out = {}
for k in sp:
    Fa = np.load(f'results/authors_ensemble_{k}.npz')['F']; Fo = np.load(f'results/our_ensemble_{k}.npz')['F']
    out[k] = dict(sr_published=sr(sp[k].values), corr_port=float(np.corrcoef(sp[k].values, Fa)[0, 1]), corr_ours=float(np.corrcoef(sp[k].values, Fo)[0, 1]))
    print(k, {a: round(b, 3) for a, b in out[k].items()})
json.dump(out, open('results/published_sdf_check.json', 'w'), indent=1)
EOF
python fill_numbers.py > /dev/null && (cd report && tectonic report.tex >/dev/null 2>&1; pdfinfo report.pdf | grep Pages)
echo "== DONE"
