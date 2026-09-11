"""
Aggregate the trained trials into the 9-model ensemble, compare with benchmarks and the paper, and draw figures.
Outputs: results/summary.json, figures/*.pdf
"""
import glob, json, os
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gan_sdf import Split, SDFNet, evaluate, normalized_factor, sharpe

os.makedirs('figures', exist_ok=True); os.makedirs('results', exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
SPL = ('train', 'valid', 'test')
S = {'train': Split('train')}
S['valid'] = Split('valid', S['train'].macro_mean, S['train'].macro_std)
S['test'] = Split('test', S['train'].macro_mean, S['train'].macro_std)
dates = pd.to_datetime(np.concatenate([S[k].dates for k in SPL]).astype(str), format='%Y%m%d')
n_tr, n_va = S['train'].T, S['valid'].T

# ---------------------------------------------------------------- our ensemble
trials = sorted(d for d in glob.glob('runs/trial_*') if os.path.exists(f'{d}/test.npz'))
print(f'{len(trials)} finished trials: {trials}')
W = {k: np.mean([np.load(f'{d}/{k}.npz')['w'] for d in trials], 0) for k in SPL}
per_trial = [{k: sharpe(normalized_factor(np.load(f'{d}/{k}.npz')['w'], S[k])[0]) for k in SPL} for d in trials]
ours = {}
for k in SPL:
    F, wn = normalized_factor(W[k], S[k]); ours[k] = F
    np.savez(f'results/our_ensemble_{k}.npz', w=W[k], w_norm=wn, F=F, dates=S[k].dates)
unc = {}
for k in SPL:  # UNC ablation = best-validation-Sharpe model of the unconditional phase (g = 1)
    pass
try:
    Wu = {k: [] for k in SPL}
    for d in trials:
        net = SDFNet(); net.load_state_dict(torch.load(f'{d}/unc_best_sharpe.pt')['sdf']); ev = evaluate(net, S)
        for k in SPL: Wu[k].append(ev[k]['w'])
    unc = {k: normalized_factor(np.mean(Wu[k], 0), S[k])[0] for k in SPL}
except Exception as e:
    print('UNC ablation skipped:', e)

auth = {k: np.load(f'results/authors_ensemble_{k}.npz')['F'] for k in SPL}
bm = np.load('results/benchmarks_factors.npz')
rows = {
    'GAN – our PyTorch retrain (9-model ensemble)': {k: sharpe(ours[k]) for k in SPL},
    'GAN – authors\' checkpoints through our port': {k: sharpe(auth[k]) for k in SPL},
    'GAN – paper Table I': dict(train=2.68, valid=1.43, test=0.75),
    'UNC (g=1) – ours': {k: sharpe(unc[k]) for k in SPL} if unc else {},
    'UNC – paper': dict(train=1.93, valid=1.33, test=0.53),
    'Ridge (EN) – ours': {k: sharpe(bm[f'Ridge_{k}']) for k in SPL},
    'EN – paper': dict(train=1.37, valid=1.15, test=0.50),
    'LS – ours': {k: sharpe(bm[f'LS_{k}']) for k in SPL},
    'LS – paper': dict(train=1.80, valid=0.58, test=0.42),
}
tf_trials = sorted(d for d in glob.glob('runs/tf_faithful_*') if os.path.exists(f'{d}/test.npz'))
if tf_trials:   # variant that mirrors four undocumented details of the TF training loop
    Wt = {k: np.mean([np.load(f'{d}/{k}.npz')['w'] for d in tf_trials], 0) for k in SPL}
    rows[f'GAN – TF-loop-faithful variant ({len(tf_trials)}-seed ensemble)'] = {k: sharpe(normalized_factor(Wt[k], S[k])[0]) for k in SPL}
    for k in SPL:
        F, wn = normalized_factor(Wt[k], S[k])
        np.savez(f'results/tf_ensemble_{k}.npz', w=Wt[k], w_norm=wn, F=F, dates=S[k].dates)
    summary_tf = [{k: sharpe(normalized_factor(np.load(f'{d}/{k}.npz')['w'], S[k])[0]) for k in SPL} for d in tf_trials]
else:
    summary_tf = []
summary = dict(sharpe=rows, per_trial=per_trial, n_trials=len(trials), per_trial_tf=summary_tf, n_trials_tf=len(tf_trials))
print(pd.DataFrame(rows).T.round(2).to_string())

# ---------------------------------------------------------------- Fig 1: cumulative SDF-factor returns
fig, ax = plt.subplots(figsize=(6.4, 3.0))
series = {'GAN (ours)': np.concatenate([ours[k] for k in SPL]), 'GAN (authors\' checkpoints)': np.concatenate([auth[k] for k in SPL]),
          'LS': np.concatenate([bm[f'LS_{k}'] for k in SPL])}
for lab, f in series.items():
    # cumulate within each sample separately and scale each sample to unit std, as in the authors' plot_SDF
    parts = [f[:n_tr], f[n_tr:n_tr + n_va], f[n_tr + n_va:]]
    cs = np.concatenate([np.cumsum(p) / p.std() for p in parts])
    ax.plot(dates, cs, lw=1.2, label=lab)
for x in (dates[n_tr], dates[n_tr + n_va]): ax.axvline(x, color='gray', ls='--', lw=0.8)
ax.text(dates[5], ax.get_ylim()[1] * 0.92, 'train', fontsize=8); ax.text(dates[n_tr + 5], ax.get_ylim()[1] * 0.92, 'valid', fontsize=8)
ax.text(dates[n_tr + n_va + 5], ax.get_ylim()[1] * 0.92, 'test (out-of-sample)', fontsize=8)
ax.set_ylabel('cumulative return\n(factor scaled to unit std per sample)'); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('figures/fig_cumret.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: Sharpe by model / trial
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={'width_ratios': [1.3, 1]})
lab = ['GAN\nours', 'GAN\nauthors', 'GAN\npaper', 'Ridge\nours', 'EN\npaper', 'LS\nours', 'LS\npaper']
keys = list(rows.keys()); keys = [keys[0], keys[1], keys[2], keys[5], keys[6], keys[7], keys[8]]
x = np.arange(len(lab)); wdt = 0.27
for j, k in enumerate(SPL):
    axes[0].bar(x + (j - 1) * wdt, [rows[q][k] for q in keys], wdt, label=k)
axes[0].set_xticks(x); axes[0].set_xticklabels(lab, fontsize=7); axes[0].set_ylabel('monthly Sharpe ratio'); axes[0].legend(frameon=False, fontsize=7)
axes[1].bar(np.arange(len(per_trial)), [p['test'] for p in per_trial], color='lightgray', label='single trial')
axes[1].axhline(rows[keys[0]]['test'], color='C0', lw=1.5, label='9-model ensemble')
axes[1].axhline(0.75, color='C2', ls='--', lw=1.2, label='paper')
axes[1].set_xlabel('trial (random seed)'); axes[1].set_ylabel('test Sharpe'); axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig('figures/fig_sharpe.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: LSTM macro states vs NBER recessions
try:
    d0 = trials[0]
    st = np.concatenate([np.load(f'{d0}/{k}.npz')['states'] for k in SPL])
    rec = pd.read_csv('data/fred/USRECM.csv'); rec.columns = ['date', 'rec']; rec['date'] = pd.to_datetime(rec['date'])
    rec = rec.set_index('date').reindex(dates.to_period('M').to_timestamp(), method='nearest')['rec'].values
    fig, axes = plt.subplots(st.shape[1], 1, figsize=(6.4, 3.4), sharex=True)
    for j, ax in enumerate(axes):
        ax.fill_between(dates, st[:, j].min(), st[:, j].max(), where=rec > 0.5, color='gray', alpha=0.3, lw=0)
        ax.plot(dates, st[:, j], lw=0.9); ax.set_ylabel(f'state {j+1}', fontsize=8)
    axes[0].set_title('LSTM hidden macro states (trial 0); shaded = NBER recessions', fontsize=9)
    fig.tight_layout(); fig.savefig('figures/fig_states.pdf'); plt.close(fig)
except Exception as e:
    print('states figure skipped:', e)

# ---------------------------------------------------------------- Fig 4: variable importance (finite-difference sensitivity)
chars = S['test'].chars
imp_auth = np.mean([np.load(f'data/sample_checkpoints/Task_1_Trial_{k}/sharpe/ave_absolute_gradient.npy') for k in range(9)], 0)
imp_ours = np.zeros(46); delta = 1e-6
with torch.no_grad():
    for d in trials:
        net = SDFNet(); net.load_state_dict(torch.load(f'{d}/best_sharpe.pt')['sdf']); net.eval()
        h = None
        for k in ('train', 'valid'): _, _, h, _ = net(S[k], h)
        te = S['test']; w0, _, _, _ = net(te, h)
        for j in range(46):
            I_bak = te.I_masked.clone(); te.I_masked[:, j] += delta
            wj, _, _, _ = net(te, h); te.I_masked = I_bak
            imp_ours[j] += (wj - w0).abs().mean().item() / delta / len(trials)
summary['variable_importance'] = dict(chars=chars, ours=imp_ours.tolist(), authors=imp_auth.tolist())
order = np.argsort(-imp_ours)[:15]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
y = np.arange(len(order))
ax.barh(y - 0.2, imp_ours[order] / imp_ours.sum(), 0.4, label='ours'); ax.barh(y + 0.2, imp_auth[order] / imp_auth.sum(), 0.4, label='authors\' checkpoints')
ax.set_yticks(y); ax.set_yticklabels([chars[i] for i in order], fontsize=8); ax.invert_yaxis(); ax.set_xlabel('normalized sensitivity of SDF weight (test sample)')
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig('figures/fig_importance.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: training curves (selection problem)
fig, ax = plt.subplots(figsize=(6.4, 2.4))
for d in trials[:3]:
    lg = json.load(open(f'{d}/log.json'))
    e_unc = [r['epoch'] for r in lg['unc']]; e_gan = [r['epoch'] + len(lg['unc']) for r in lg['gan']]
    ax.plot(e_unc + e_gan, [r['sr_train'] for r in lg['unc'] + lg['gan']], color='C0', lw=0.8, alpha=0.7)
    ax.plot(e_unc + e_gan, [r['sr_valid'] for r in lg['unc'] + lg['gan']], color='C1', lw=0.8, alpha=0.7)
ax.axvline(len(lg['unc']), color='gray', ls='--', lw=0.8); ax.text(len(lg['unc']) + 5, ax.get_ylim()[1] * 0.9, 'adversary g introduced', fontsize=7)
ax.plot([], [], color='C0', label='train Sharpe'); ax.plot([], [], color='C1', label='validation Sharpe'); ax.legend(frameon=False, fontsize=8)
ax.set_xlabel('epoch (3 trials shown)'); ax.set_ylabel('monthly SR'); fig.tight_layout(); fig.savefig('figures/fig_training.pdf'); plt.close(fig)

json.dump(summary, open('results/summary.json', 'w'), indent=1)
print('figures written:', sorted(glob.glob('figures/*.pdf')))
