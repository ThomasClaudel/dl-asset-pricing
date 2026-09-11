"""
Second stage of Chen-Pelger-Zhu: risk loadings beta_{t,i} ∝ E_t[R^e_{t+1,i} F_{t+1}] estimated with a feed-forward
network (3 layers [32,16,8], 46 characteristics, no macro; the authors' config_RF_1.json) that predicts
R^e_{t+1,i} * F_{t+1} * 50.  Beta then gives the explained variation (EV) and cross-sectional R^2 of Table I via a
per-month projection of returns on beta (calculate_statistics).
usage: python beta_ffn.py <prefix>   where <prefix>_{train,valid,test}.npz contain the SDF factor F
"""
import sys, json, os
import numpy as np, torch, torch.nn as nn
from gan_sdf import Split, calculate_statistics, _init_tf_like

prefix = sys.argv[1]
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DEV = sys.argv[3] if len(sys.argv) > 3 else ('mps' if torch.backends.mps.is_available() else 'cpu')
EPOCHS, SCALE = 2048, 50.0
torch.set_num_threads(2)
S = {k: Split(k, device='cpu') for k in ('train',)}
S['valid'] = Split('valid', S['train'].macro_mean, S['train'].macro_std)
S['test'] = Split('test', S['train'].macro_mean, S['train'].macro_std)
F = {k: np.load(f'{prefix}_{k}.npz')['F'] for k in S}
X = {k: S[k].I_masked.to(DEV) for k in S}
Y = {k: torch.tensor(((S[k].R_np * F[k][:, None]) * SCALE)[S[k].mask_np], dtype=torch.float32, device=DEV) for k in S}


def make():
    net = nn.Sequential(nn.Linear(46, 32), nn.ReLU(), nn.Dropout(0.05), nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.05),
                        nn.Linear(16, 8), nn.ReLU(), nn.Dropout(0.05), nn.Linear(8, 1))
    net.apply(_init_tf_like); return net


betas = {k: [] for k in S}
for seed in range(n_seeds):
    torch.manual_seed(seed)
    net = make().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_sd = np.inf, None
    for ep in range(EPOCHS):
        net.train(); opt.zero_grad()
        loss = ((net(X['train']).squeeze(1) - Y['train']) ** 2).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(X['valid']).squeeze(1) - Y['valid']) ** 2).mean().item()
        if vl < best: best, best_sd = vl, {a: b.clone() for a, b in net.state_dict().items()}
    net.load_state_dict(best_sd); net.eval()
    with torch.no_grad():
        for k in S: betas[k].append(net(X[k]).squeeze(1).cpu().numpy())
    print(f'seed {seed}: best valid mse {best:.4f}', flush=True)

res = {}
for k in S:
    ev, xs, xsw = calculate_statistics(np.mean(betas[k], 0), S[k])
    res[k] = dict(EV=ev, XSR2=xs, XSR2_weighted=xsw)
    print(f'{k:5s}  EV {ev:.3f}   XS-R2 {xs:.3f}   XS-R2 (weighted, Table I) {xsw:.3f}')
print('paper GAN Table I:  EV train/valid/test 0.20/0.09/0.08   XS-R2 0.12/0.01/0.23')
json.dump(res, open(f'{prefix}_beta_stats.json', 'w'), indent=1)
np.savez(f'{prefix}_beta.npz', **{k: np.mean(betas[k], 0) for k in S})
