"""
PyTorch re-implementation of Chen, Pelger & Zhu (2024), "Deep Learning in Asset Pricing".

Mirrors the authors' TF-1.12 code (github.com/jasonzy121/Deep_Learning_Asset_Pricing):
  * SDF network   : LSTM(178 -> 4 macro states) + FFN([46 chars, 4 states] -> 64 -> 64 -> 1) = omega_{t,i}
                    M_{t+1} = 1 + sum_i omega_{t,i} R^e_{t+1,i}          (SDF factor F = 1 - M = -omega'R)
  * moment network: LSTM(178 -> 32) + linear([46, 32] -> 8, tanh) = g_{t,i}  (adversarial test assets)
  * loss          : mean_{j,i} T_i/max T_i * ( 1/T_i sum_t M_{t+1} R^e_{t+1,i} g_{j,t,i} )^2
  * training      : (1) 256 epochs unconditional (g = 1), (2) 64 steps of adversary on best-valid-loss SDF,
                    (3) 1024 epochs conditional; 4 gradient steps per epoch on the full batch; Adam 1e-3;
                    dropout keep 0.95; model selection = best validation Sharpe.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

UNK = -99.99
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'datasets')


# ----------------------------------------------------------------------------- data
class Split:
    def __init__(self, name, macro_mean=None, macro_std=None, use_macro=True, device='cpu'):
        z = np.load(f'{DATA}/char/Char_{name}.npz', allow_pickle=True)
        d = z['data'].astype(np.float32)
        self.dates = z['date']
        self.chars = [str(v) for v in z['variable'][1:]]
        R = d[:, :, 0]
        self.mask_np = R != UNK
        self.R_np = np.where(self.mask_np, R, 0.0).astype(np.float32)
        self.I_np = d[:, :, 1:]
        self.T, self.N = R.shape
        self.T_i = self.mask_np.sum(0).astype(np.float32)   # obs per stock  (loss weights)
        self.N_t = self.mask_np.sum(1)                       # stocks per month

        m = np.load(f'{DATA}/macro/macro_{name}.npz', allow_pickle=True)
        M = m['data'].astype(np.float32)
        self.macro_names = [str(v) for v in m['variable']]
        if macro_mean is None:
            macro_mean, macro_std = M.mean(0), M.std(0)
        self.macro_mean, self.macro_std = macro_mean, macro_std
        M = (M - macro_mean) / macro_std
        if not use_macro:
            M = np.zeros((self.T, 0), dtype=np.float32)
        self.M_np = M

        dev = torch.device(device)
        self.R = torch.tensor(self.R_np, device=dev)
        self.mask = torch.tensor(self.mask_np, device=dev)
        self.I = torch.tensor(self.I_np, device=dev)
        self.M = torch.tensor(self.M_np, device=dev)
        self.Tw = torch.tensor(self.T_i, device=dev)
        self.I_masked = self.I[self.mask]                    # (n_obs, 46)
        self.t_index = torch.arange(self.T, device=dev).unsqueeze(1).expand(self.T, self.N)[self.mask]


# ----------------------------------------------------------------------------- networks
def _init_tf_like(m):
    """glorot-uniform kernels, zero bias (TF Dense / LSTMCell defaults); LSTM forget bias = 1."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    if isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if 'weight' in n: nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
                H = m.hidden_size
                p.data[H:2 * H] = 0.5      # two bias vectors in torch -> forget bias sums to 1.0


class SDFNet(nn.Module):
    def __init__(self, n_char=46, n_macro=178, n_state=4, hidden=(64, 64), p_drop=0.05):
        super().__init__()
        self.n_state = n_state if n_macro > 0 else 0
        self.lstm = nn.LSTM(n_macro, n_state, batch_first=True) if n_macro > 0 else None
        self.in_drop = nn.Dropout(p_drop)
        layers, d = [], n_char + self.n_state
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]; d = h
        layers += [nn.Linear(d, 1)]
        self.ffn = nn.Sequential(*layers)
        self.apply(_init_tf_like)

    def states(self, M, h0=None):
        if self.lstm is None:
            return None, None
        out, hT = self.lstm(self.in_drop(M).unsqueeze(0), h0)
        return out.squeeze(0), hT                             # (T, n_state), final state

    def forward(self, split, h0=None):
        s, hT = self.states(split.M, h0)
        x = split.I_masked if s is None else torch.cat([split.I_masked, s[split.t_index]], 1)
        w = self.ffn(x).squeeze(1)                            # (n_obs,)
        wR = torch.zeros_like(split.R).masked_scatter(split.mask, w * split.R[split.mask])
        sdf = 1.0 + wR.sum(1)                                 # (T,)
        return w, sdf, hT, s


class MomentNet(nn.Module):
    def __init__(self, n_char=46, n_macro=178, n_state=32, n_cond=8, p_drop=0.05):
        super().__init__()
        self.lstm = nn.LSTM(n_macro, n_state, batch_first=True) if n_macro > 0 else None
        self.in_drop = nn.Dropout(p_drop)
        self.out = nn.Linear(n_char + (n_state if n_macro > 0 else 0), n_cond)
        self.apply(_init_tf_like)

    def forward(self, split):
        if self.lstm is None:
            x = split.I
        else:
            s, _ = self.lstm(self.in_drop(split.M).unsqueeze(0))
            s = s.squeeze(0).unsqueeze(1).expand(-1, split.N, -1)
            x = torch.cat([s, split.I], 2)
        return torch.tanh(self.out(x)).permute(2, 0, 1)       # (n_cond, T, N)


# ----------------------------------------------------------------------------- loss / metrics
def moment_loss(sdf, split, g):
    """g: (J, T, N) or scalar 1.  weighted by T_i / max T_i as in the authors' code."""
    Rm = split.R * split.mask
    emp = (Rm.unsqueeze(0) * sdf.view(1, -1, 1) * g).sum(1) / split.Tw    # (J, N)
    return (emp.pow(2) * (split.Tw / split.Tw.max())).mean()


def normalized_factor(w, split):
    """SDF factor F = -sum_i w_norm R with per-month L1-normalized weights (authors' reported numbers)."""
    w = np.asarray(w, dtype=np.float64)
    parts = np.split(w, np.cumsum(split.N_t)[:-1])
    wn = np.concatenate([p / np.abs(p).sum() for p in parts])
    wR = np.zeros(split.mask_np.shape); wR[split.mask_np] = wn * split.R_np[split.mask_np]
    return -wR.sum(1), wn


def sharpe(f):
    return float(np.mean(f) / np.std(f))


def calculate_statistics(beta, split):
    """EV, XS-R2, weighted XS-R2 -- verbatim port of model_utils.calculateStatistics."""
    mask, R = split.mask_np, split.R_np.astype(np.float64)
    parts = np.cumsum(split.N_t)[:-1]
    b_list, R_list = np.split(np.asarray(beta, np.float64), parts), np.split(R[mask], parts)
    res = np.zeros(mask.shape)
    res_l = [R_i - (b_i @ R_i) / (b_i @ b_i) * b_i for R_i, b_i in zip(R_list, b_list)]
    res[mask] = np.concatenate(res_l)
    T_i, N_t = mask.sum(0), mask.sum(1)
    Rm = R * mask
    ev = 1 - np.mean((res ** 2).sum(1) / N_t) / np.mean((Rm ** 2).sum(1) / N_t)
    xs = 1 - np.mean((res.sum(0) / T_i) ** 2) / np.mean((Rm.sum(0) / T_i) ** 2)
    xsw = 1 - np.mean((res.sum(0) / T_i) ** 2 * T_i) / np.mean((Rm.sum(0) / T_i) ** 2 * T_i)
    return ev, xs, xsw


# ----------------------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, splits, names=('train', 'valid', 'test')):
    """Run train -> valid -> test carrying the LSTM state; returns per-split (w, sdf, states)."""
    model.eval()
    out, h = {}, None
    for name in names:
        w, sdf, h, s = model(splits[name], h)
        out[name] = dict(w=w.cpu().numpy(), sdf=sdf.cpu().numpy(),
                         states=None if s is None else s.cpu().numpy())
    model.train()
    return out


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    dev = args.device
    tr = Split('train', use_macro=not args.no_macro, device=dev)
    splits = dict(train=tr,
                  valid=Split('valid', tr.macro_mean, tr.macro_std, not args.no_macro, dev),
                  test=Split('test', tr.macro_mean, tr.macro_std, not args.no_macro, dev))
    n_macro = tr.M.shape[1]
    sdf_net = SDFNet(n_macro=n_macro, n_state=args.n_state, p_drop=args.dropout).to(dev)
    mom_net = MomentNet(n_macro=n_macro, n_state=args.n_state_moment, n_cond=args.n_cond, p_drop=args.dropout).to(dev)
    opt_sdf = torch.optim.Adam(sdf_net.parameters(), lr=args.lr)
    opt_mom = torch.optim.Adam(mom_net.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    log = dict(config=vars(args), unc=[], gan=[], moment=[])

    def val_metrics(with_test=False):
        # test split is only evaluated for logging (never for selection) -- skip it most epochs for speed
        ev = evaluate(sdf_net, splits, ('train', 'valid', 'test') if with_test else ('train', 'valid'))
        with torch.no_grad():
            lv = moment_loss(torch.tensor(ev['valid']['sdf'], device=dev), splits['valid'], 1.0).item()
        sr = {k: sharpe(normalized_factor(ev[k]['w'], splits[k])[0]) for k in ev}
        sr.setdefault('test', float('nan'))
        # the authors' training loop selects on the Sharpe of the *unnormalized* factor 1 - SDF (getSDF), while the
        # notebook reports the L1-normalized one; keep both so either can drive checkpoint selection
        sr['valid_unnorm'] = sharpe(1.0 - ev['valid']['sdf'])
        sr['select'] = sr['valid_unnorm'] if args.select_unnorm else sr['valid']
        return lv, sr

    best = dict(loss=(np.inf, None), sharpe=(-np.inf, None))

    def maybe_save(epoch, lv, sr, phase):
        if epoch <= args.ignore_epoch: return
        if lv < best['loss'][0]:
            best['loss'] = (lv, {k: v.clone() for k, v in sdf_net.state_dict().items()})
        if sr['select'] > best['sharpe'][0]:
            best['sharpe'] = (sr['select'], {k: v.clone() for k, v in sdf_net.state_dict().items()})
            torch.save(dict(sdf=sdf_net.state_dict(), mom=mom_net.state_dict(), epoch=epoch, phase=phase, sr=sr),
                       f'{args.out}/best_sharpe.pt')

    # ---- phase 1: unconditional moments (g = 1)
    t0 = time.time()
    for ep in range(args.epochs_unc):
        for _ in range(args.sub_epoch):
            opt_sdf.zero_grad()
            _, sdf, _, _ = sdf_net(tr)
            loss = moment_loss(sdf, tr, 1.0)
            loss.backward(); opt_sdf.step()
        lv, sr = val_metrics(ep % args.print_every == 0); maybe_save(ep, lv, sr, 'unc')
        log['unc'].append(dict(epoch=ep, train_loss=loss.item(), valid_loss=lv, **{f'sr_{k}': v for k, v in sr.items()}))
        if ep % args.print_every == 0:
            print(f'[UNC {ep:4d}] loss {loss.item():.3e} vloss {lv:.3e} SR tr/va/te {sr["train"]:.2f}/{sr["valid"]:.2f}/{sr["test"]:.2f}  {time.time()-t0:.0f}s', flush=True)
    log['unc_best_sharpe'] = best['sharpe'][0]
    torch.save(dict(sdf=best['sharpe'][1]), f'{args.out}/unc_best_sharpe.pt')

    # ---- phase 2: adversary maximizes the conditional loss on the best-valid-loss SDF
    sdf_net.load_state_dict(best['loss'][1])
    with torch.no_grad():
        _, sdf_fixed, _, _ = sdf_net.eval()(tr); sdf_net.train()
    best_mom, best_mom_loss = None, -np.inf
    for ep in range(args.epochs_moment):
        opt_mom.zero_grad()
        g = mom_net(tr)
        if args.stochastic_g:   # TF feeds dropout=0.95 here too: the adversary sees a dropout-perturbed SDF
            with torch.no_grad():
                _, sdf_fixed, _, _ = sdf_net(tr)
        loss = moment_loss(sdf_fixed, tr, g)
        if loss.item() > best_mom_loss:
            best_mom_loss, best_mom = loss.item(), {k: v.clone() for k, v in mom_net.state_dict().items()}
        (-loss).backward(); opt_mom.step()
        log['moment'].append(dict(epoch=ep, loss=loss.item()))
    mom_net.load_state_dict(best_mom)
    print(f'[MOMENT] adversarial loss raised to {best_mom_loss:.3e}', flush=True)
    with torch.no_grad():
        mom_net.eval(); g_fixed = mom_net(tr).detach(); mom_net.train()

    # ---- phase 3: conditional moments with adversarial test assets g (fixed, or re-sampled with dropout as in TF)
    best['loss'] = (np.inf, None)
    if not args.keep_unc_best:      # TF never resets its best-validation-Sharpe tracker between phases
        best['sharpe'] = (-np.inf, None)
    if args.fresh_adam:             # TF builds a separate Adam op (fresh moments) for the conditional phase
        opt_sdf = torch.optim.Adam(sdf_net.parameters(), lr=args.lr)
    t0 = time.time()
    for ep in range(args.epochs_gan):
        for _ in range(args.sub_epoch):
            opt_sdf.zero_grad()
            _, sdf, _, _ = sdf_net(tr)
            if args.stochastic_g:
                with torch.no_grad():
                    g_step = mom_net(tr)   # train mode: dropout on the adversary's LSTM input, as in TF
            else:
                g_step = g_fixed
            loss = moment_loss(sdf, tr, g_step)
            loss.backward(); opt_sdf.step()
        lv, sr = val_metrics(ep % args.print_every == 0); maybe_save(ep, lv, sr, 'gan')
        log['gan'].append(dict(epoch=ep, train_loss=loss.item(), valid_loss=lv, **{f'sr_{k}': v for k, v in sr.items()}))
        if ep % args.print_every == 0:
            print(f'[GAN {ep:4d}] loss {loss.item():.3e} vloss {lv:.3e} SR tr/va/te {sr["train"]:.2f}/{sr["valid"]:.2f}/{sr["test"]:.2f}  {time.time()-t0:.0f}s', flush=True)

    # ---- final: best-validation-Sharpe model -> weights, factor, states
    sdf_net.load_state_dict(best['sharpe'][1])
    ev = evaluate(sdf_net, splits)
    res = {}
    for k in ev:
        F, wn = normalized_factor(ev[k]['w'], splits[k])
        res[k] = dict(sr=sharpe(F))
        np.savez(f'{args.out}/{k}.npz', w=ev[k]['w'], w_norm=wn, F=F, sdf=ev[k]['sdf'],
                 states=ev[k]['states'] if ev[k]['states'] is not None else np.zeros(0), dates=splits[k].dates)
    log['final'] = res
    print('FINAL (best valid Sharpe)  SR train/valid/test: %.3f / %.3f / %.3f' % (res['train']['sr'], res['valid']['sr'], res['test']['sr']), flush=True)
    json.dump(log, open(f'{args.out}/log.json', 'w'), indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/trial_0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--epochs_unc', type=int, default=256)
    p.add_argument('--epochs_moment', type=int, default=64)
    p.add_argument('--epochs_gan', type=int, default=1024)
    p.add_argument('--sub_epoch', type=int, default=4)
    p.add_argument('--ignore_epoch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.05)
    p.add_argument('--n_state', type=int, default=4)
    p.add_argument('--n_state_moment', type=int, default=32)
    p.add_argument('--n_cond', type=int, default=8)
    p.add_argument('--no_macro', action='store_true')
    p.add_argument('--stochastic_g', action='store_true', help='re-sample adversary instruments with dropout every step (TF behaviour)')
    p.add_argument('--select_unnorm', action='store_true', help='select checkpoints on the unnormalized-factor validation Sharpe (TF behaviour)')
    p.add_argument('--keep_unc_best', action='store_true', help='do not reset the best-validation-Sharpe tracker between phases (TF behaviour)')
    p.add_argument('--fresh_adam', action='store_true', help='new Adam state for the conditional phase (TF behaviour)')
    p.add_argument('--print_every', type=int, default=32)
    run(p.parse_args())
