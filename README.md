# Replicating *Deep Learning in Asset Pricing* (Chen, Pelger & Zhu, 2024)

A from-scratch PyTorch replication of the GAN stochastic discount factor of
Chen, Pelger & Zhu, *Management Science* 70(2), 2024 ([arXiv:1904.00745](https://arxiv.org/abs/1904.00745)),
done as the final project for MFE 230ZA (ML for Finance 2: Control, Action and Reinforcement Learning), UC Berkeley Haas.

The 2-page write-up (data, method, challenges and how they were solved, extensions) is
[`report/report.pdf`](report/report.pdf).

## Headline results (out-of-sample 1992–2016, monthly Sharpe ratio of the SDF factor)

| Model | Paper (Table I) | This replication |
|---|---|---|
| GAN — retrained from scratch, 9-seed ensemble | 0.75 | **0.66** |
| GAN — variant mirroring the TF training loop, 3-seed ensemble | 0.75 | **0.73** |
| GAN — authors' released checkpoints run through this code | 0.75 | **0.77** |
| UNC (unconditional moments, no adversary) | 0.53 | 0.45 |
| EN / ridge | 0.50 | 0.49 |
| LS (linear, closed form) | 0.42 | **0.42** |
| GAN explained variation / cross-sectional R² | 0.08 / 0.23 | 0.06 / 0.22 |

The authors' published SDF series (`Data.xlsx`) reproduces Table I to three decimals; the ported checkpoints
correlate 0.87 (test) with it, our retrain 0.83.

## What is in here

| File | Purpose |
|---|---|
| `gan_sdf.py` | The model: LSTM macro-state + FFN SDF network, adversarial moment network, three-phase training, Sharpe / EV / XS-R² metrics. Flags `--stochastic_g --select_unnorm --keep_unc_best --fresh_adam` reproduce four undocumented details of the authors' TF loop. |
| `verify_checkpoints.py` | Loads the authors' TF-1.12 checkpoints into the PyTorch model (LSTM gate re-ordering, forget-gate bias) and recomputes the ensemble Sharpe. |
| `benchmarks.py` | LS (closed form on 92 long/short characteristic-managed portfolios) and ridge benchmarks. |
| `beta_ffn.py` | Second-stage β network (46→32→16→8→1 on R·F) giving EV and XS-R². |
| `extensions.py` | Turnover (paper's Table A.VII definition), Sharpe net of transaction costs, sub-period Sharpe, drawdowns. |
| `make_figures.py` | Ensemble aggregation, comparison table, all figures. |
| `export_factors.py` | Writes the monthly SDF-factor series of every model to `results/sdf_factors.csv`. |
| `fill_numbers.py`, `report/report.tex` | The write-up; every number is generated from `results/*.json`, none hand-typed. |
| `finalize.sh`, `finalize2.sh` | The full downstream chain after training. |
| `runs/trial_*/`, `runs/tf_faithful_*/` | Trained checkpoints (`best_sharpe.pt`), unconditional-phase checkpoints, per-epoch logs. |
| `results/` | All summary numbers (`summary.json`, `benchmarks.json`, `*_beta_stats.json`, `*_extensions.json`, `published_sdf_check.json`) and `sdf_factors.csv`. |
| `figures/` | Cumulative returns, Sharpe by model/seed, LSTM macro states vs NBER recessions, variable importance, training curves. |

## Data

The panel is **CRSP monthly excess returns with 46 firm characteristics (rank-normalized to [−0.5, 0.5]) and 178
macro series, exactly as released by the paper's authors** — not Yahoo, not rebuilt. It is not in this repository
(3.8 GB, CRSP-derived). Get it from the Google Drive folder linked on
[Markus Pelger's data-and-code page](https://mpelger.people.stanford.edu/data-and-code):

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l" -O data
cd data && unzip datasets.zip && unzip sample_checkpoints.zip
```

Also used: `USRECM` from FRED (recession shading) and the authors' `Data.xlsx` (published SDF series,
Dropbox link on the same page), both placed under `data/`.

## Reproduce

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install numpy pandas torch matplotlib scipy gdown openpyxl tensorflow   # tensorflow only for verify_checkpoints.py

python benchmarks.py                                   # LS / ridge, seconds
python verify_checkpoints.py                           # authors' checkpoints through this code
for s in 0 1 2 3 4 5 6 7 8; do python gan_sdf.py --device mps --out runs/trial_$s --seed $s; done   # ~12 min each on Apple GPU, ~35 min CPU
for s in 10 11 12; do python gan_sdf.py --device mps --out runs/tf_faithful_$s --seed $s \
      --stochastic_g --select_unnorm --keep_unc_best --fresh_adam; done
./finalize.sh && ./finalize2.sh                        # ensemble, figures, extensions, beta stage, report
```

Model, hyper-parameters and schedule are the authors' optimal configuration (`config/config.json` in their repository):
2×64 SDF layers, 4 LSTM macro states, 8 adversarial instruments with a 32-state LSTM, dropout 0.05, Adam 1e-3,
256 unconditional epochs → 64 adversary steps → 1024 conditional epochs, 4 full-batch steps per epoch,
best-validation-Sharpe checkpoint, ensemble of seeds.

## References

- Chen, L., Pelger, M., Zhu, J. (2024). Deep Learning in Asset Pricing. *Management Science* 70(2), 714–750.
- Official code (TensorFlow 1.12): https://github.com/jasonzy121/Deep_Learning_Asset_Pricing
