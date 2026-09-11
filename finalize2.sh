#!/bin/zsh
# After the TF-loop-faithful variant seeds finish: fold them into the summary/figures, run their beta stage, rebuild.
set -e
cd "$(dirname "$0")"
. .venv/bin/activate
echo "== ensemble + figures (default + variant)"; python make_figures.py 2>&1 | grep -v -E 'WARNING|absl|oneDNN'
echo "== variant extensions";                     python extensions.py results/tf_ensemble
echo "== variant beta network (MPS)";             python beta_ffn.py results/tf_ensemble 3 mps
echo "== numbers.tex + report";                   python fill_numbers.py > /dev/null
(cd report && tectonic report.tex 2>&1 | grep -E -i 'error|undefined' || true; pdfinfo report.pdf | grep Pages)
echo "== DONE"
