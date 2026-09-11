"""Write the monthly SDF-factor series of every model to one small CSV (results/sdf_factors.csv)."""
import numpy as np, pandas as pd
from gan_sdf import Split

SPL = ('train', 'valid', 'test')
dates = pd.to_datetime(np.concatenate([Split(k).dates for k in SPL]).astype(str), format='%Y%m%d')
cols = {'sample': np.concatenate([[k] * len(Split(k).dates) for k in SPL])}
for name, prefix in [('GAN_ours_default_9seed', 'results/our_ensemble'), ('GAN_ours_tf_faithful', 'results/tf_ensemble'),
                     ('GAN_authors_checkpoints_ported', 'results/authors_ensemble')]:
    try:
        cols[name] = np.concatenate([np.load(f'{prefix}_{k}.npz')['F'] for k in SPL])
    except FileNotFoundError:
        pass
bm = np.load('results/benchmarks_factors.npz')
cols['LS'] = np.concatenate([bm[f'LS_{k}'] for k in SPL]); cols['Ridge'] = np.concatenate([bm[f'Ridge_{k}'] for k in SPL])
df = pd.DataFrame(cols, index=dates.rename('date'))
df.to_csv('results/sdf_factors.csv', float_format='%.6f')
print(df.groupby('sample').agg(lambda s: s.mean() / s.std()).round(3).T)
