"""
Cheap extensions on top of the paper's results, computed from the ensemble SDF weights / factor:
  1. turnover of the SDF portfolio (authors' calculateTurnover definition) and Sharpe net of proportional
     transaction costs of 0/10/25/50 bp per unit of turnover
  2. sub-period out-of-sample Sharpe (1992-2004 vs 2005-2016) and drawdown statistics
usage: python extensions.py <prefix>     (prefix_{train,valid,test}.npz with w_norm and F)
"""
import sys, json
import numpy as np, pandas as pd
from gan_sdf import Split, sharpe

prefix = sys.argv[1]
S = {'train': Split('train')}
S['valid'] = Split('valid', S['train'].macro_mean, S['train'].macro_std)
S['test'] = Split('test', S['train'].macro_mean, S['train'].macro_std)


def turnover(sp, w_norm):
    """Port of src/utils.calculateTurnover with Rf=0 (excess returns): long and short legs separately."""
    w = np.zeros(sp.mask_np.shape); w[sp.mask_np] = -w_norm          # SDF *portfolio* weights are -omega
    w = np.concatenate([w, 1 - w.sum(1, keepdims=True)], 1)          # residual in the risk-free asset
    R = np.concatenate([sp.R_np, np.zeros((sp.T, 1))], 1)
    wp, wm = np.maximum(w, 0), -np.minimum(w, 0)
    Rp, Rm = (R * wp).sum(1, keepdims=True), (R * wm).sum(1, keepdims=True)
    Tp = np.abs((1 + Rp)[:-1] * wp[1:] - ((1 + R) * wp)[:-1]).sum(1) / wp[:-1].sum(1)
    Tm = np.abs((1 + Rm)[:-1] * wm[1:] - ((1 + R) * wm)[:-1]).sum(1) / wm[:-1].sum(1)
    return Tp, Tm


out = {}
for k in ('valid', 'test'):
    z = np.load(f'{prefix}_{k}.npz'); F, wn = z['F'], z['w_norm']
    Tp, Tm = turnover(S[k], wn)
    # total dollar turnover per month relative to the L1 norm (=1) of the SDF weights
    w = np.zeros(S[k].mask_np.shape); w[S[k].mask_np] = wn
    dollar_turn = np.abs(np.diff(w, axis=0)).sum(1)                   # approximate: |w_t - w_{t-1}| summed
    res = dict(turnover_long=float(Tp.mean()), turnover_short=float(Tm.mean()), dollar_turnover=float(dollar_turn.mean()),
               sr_gross=sharpe(F), max_1m_loss=float(F.min()), std_monthly=float(F.std()))
    for bp in (10, 25, 50):
        Fnet = F.copy(); Fnet[1:] -= bp / 1e4 * dollar_turn
        res[f'sr_net_{bp}bp'] = sharpe(Fnet)
    if k == 'test':
        d = pd.to_datetime(S[k].dates.astype(str), format='%Y%m%d')
        res['sr_1992_2004'] = sharpe(F[d.year <= 2004]); res['sr_2005_2016'] = sharpe(F[d.year >= 2005])
        res['sr_excl_2008_2009'] = sharpe(F[(d.year < 2008) | (d.year > 2009)])
        cum = np.cumsum(F); res['max_drawdown_cum'] = float((np.maximum.accumulate(cum) - cum).max())
    out[k] = res
    print(k, json.dumps({a: round(b, 3) for a, b in res.items()}))
json.dump(out, open(f'{prefix}_extensions.json', 'w'), indent=1)
