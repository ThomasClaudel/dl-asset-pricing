"""
Linear benchmarks from Chen-Pelger-Zhu Table I:
  LS  : SDF weights omega = theta' I with theta = (E[F~ F~'])^{-1} E[F~], F~ = characteristic-managed portfolios
        (positive and negative leg of each of the 46 rank-characteristics separately -> 92 factors, as in the paper)
  EN  : we use the closed-form ridge special case theta_lambda = (E[F~F~'] + lambda I)^{-1} E[F~], lambda picked on
        validation Sharpe (the paper adds an L1 term too; ridge is the part with a closed form).
Also reports EV / XS-R2 with a linear beta (OLS of R*F on I over the training set).
"""
import json, os
import numpy as np
from gan_sdf import Split, sharpe, calculate_statistics

os.makedirs('results', exist_ok=True)
S = {k: Split(k, device='cpu') for k in ('train', 'valid', 'test')}


def legs(I):                                   # (…,46) -> (…,92): long leg (I>0) and short leg (I<0)
    return np.concatenate([np.maximum(I, 0), np.minimum(I, 0)], -1)


def managed_portfolios(sp):                    # F~_{t+1} = 1/N_t sum_i legs(I_{t,i}) R_{t+1,i}
    X = legs(sp.I_np) * sp.mask_np[..., None]
    return np.einsum('tnk,tn->tk', X, sp.R_np) / sp.N_t[:, None]


Ft = {k: managed_portfolios(S[k]) for k in S}
Sigma = Ft['train'].T @ Ft['train'] / Ft['train'].shape[0]
mu = Ft['train'].mean(0)


def factor(theta, k):                          # SDF factor F_{t+1} = theta' F~_{t+1}
    return Ft[k] @ theta


def linear_beta_stats(theta):
    """beta_{t,i} = b' I_{t,i}, b = OLS of R_{t+1,i} F_{t+1} on legs(I_{t,i}) pooled over the training sample."""
    tr = S['train']; F = factor(theta, 'train')
    X = legs(tr.I_np[tr.mask_np]); y = (tr.R_np * F[:, None])[tr.mask_np]
    X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    out = {}
    for k in S:
        Xk = np.concatenate([legs(S[k].I_np[S[k].mask_np]), np.ones((S[k].mask_np.sum(), 1))], 1)
        ev, xs, xsw = calculate_statistics(Xk @ b, S[k])
        out[k] = dict(EV=ev, XSR2=xs, XSR2_weighted=xsw)
    return out


res = {}
theta_ls = np.linalg.solve(Sigma, mu)
res['LS'] = dict(SR={k: sharpe(factor(theta_ls, k)) for k in S}, stats=linear_beta_stats(theta_ls))
print('LS   SR train/valid/test = %.2f / %.2f / %.2f   (paper: 1.80 / 0.58 / 0.42)' % tuple(res['LS']['SR'].values()))

best = None
for lam in np.logspace(-8, -2, 25):
    th = np.linalg.solve(Sigma + lam * np.eye(len(mu)), mu)
    sr = {k: sharpe(factor(th, k)) for k in S}
    if best is None or sr['valid'] > best[1]['valid']:
        best = (lam, sr, th)
res['Ridge'] = dict(lam=float(best[0]), SR=best[1], stats=linear_beta_stats(best[2]))
print('Ridge SR train/valid/test = %.2f / %.2f / %.2f  lambda=%.1e  (paper EN: 1.37 / 1.15 / 0.50)' % (*best[1].values(), best[0]))
for m in res:
    s = res[m]['stats']['test']
    print(f'{m:5s} test EV {s["EV"]:.3f}  XS-R2 {s["XSR2"]:.3f}  XS-R2(w) {s["XSR2_weighted"]:.3f}')
np.savez('results/benchmarks_factors.npz', LS_train=factor(theta_ls, 'train'), LS_valid=factor(theta_ls, 'valid'),
         LS_test=factor(theta_ls, 'test'), Ridge_train=factor(best[2], 'train'), Ridge_valid=factor(best[2], 'valid'),
         Ridge_test=factor(best[2], 'test'))
json.dump(res, open('results/benchmarks.json', 'w'), indent=1)
