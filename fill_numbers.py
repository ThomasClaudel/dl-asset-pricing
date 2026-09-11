"""Write report/numbers.tex from results/*.json so the LaTeX report never contains hand-typed numbers."""
import json
s = json.load(open('results/summary.json'))['sharpe']
b = json.load(open('results/benchmarks.json'))
def opt(path):
    try: return json.load(open(path))
    except FileNotFoundError: return None
auth_beta = opt('results/authors_ensemble_beta_stats.json') or {k: dict(EV=float('nan'), XSR2_weighted=float('nan')) for k in ('train', 'valid', 'test')}
our_beta = opt('results/our_ensemble_beta_stats.json')
try: ext = json.load(open('results/our_ensemble_extensions.json'))
except FileNotFoundError: ext = json.load(open('results/authors_ensemble_extensions.json'))
per = json.load(open('results/summary.json'))['per_trial']
n = json.load(open('results/summary.json'))['n_trials']

def f2(x): return '%.2f' % x
L = []
def m(name, val): L.append(r'\newcommand{\%s}{%s}' % (name, val))
ours = s['GAN – our PyTorch retrain (9-model ensemble)']; auth = s["GAN – authors' checkpoints through our port"]
for k, tag in (('train', 'Tr'), ('valid', 'Va'), ('test', 'Te')):
    m('ganSR' + tag, f2(ours[k])); m('authSR' + tag, f2(auth[k]))
    m('lsSR' + tag, f2(s['LS – ours'][k])); m('ridgeSR' + tag, f2(s['Ridge (EN) – ours'][k]))
    if s['UNC (g=1) – ours']: m('uncSR' + tag, f2(s['UNC (g=1) – ours'][k]))
    else: m('uncSR' + tag, '--')
sm = json.load(open('results/summary.json'))
tfrow = next((v for k, v in s.items() if k.startswith('GAN – TF-loop-faithful')), None)
if tfrow and sm.get('per_trial_tf'):
    for k, tag in (('train', 'Tr'), ('valid', 'Va'), ('test', 'Te')): m('tfSR' + tag, f2(tfrow[k]))
    m('nTrialsTf', str(sm['n_trials_tf']))
    m('tfTeMin', f2(min(p['test'] for p in sm['per_trial_tf']))); m('tfTeMax', f2(max(p['test'] for p in sm['per_trial_tf'])))
    m('tfVaMin', f2(min(p['valid'] for p in sm['per_trial_tf']))); m('tfVaMax', f2(max(p['valid'] for p in sm['per_trial_tf'])))
else:
    for q in ('tfSRTr', 'tfSRVa', 'tfSRTe', 'nTrialsTf', 'tfTeMin', 'tfTeMax', 'tfVaMin', 'tfVaMax'): m(q, '--')
m('trialVaMin', f2(min(p['valid'] for p in per))); m('trialVaMax', f2(max(p['valid'] for p in per)))
m('nTrials', str(n)); m('ganSRTeAnn', '%.1f' % (ours['test'] * 12 ** 0.5)); m('authSRTeAnn', '%.1f' % (auth['test'] * 12 ** 0.5))
m('trialTeMin', f2(min(p['test'] for p in per))); m('trialTeMax', f2(max(p['test'] for p in per)))
m('authEVTe', f2(auth_beta['test']['EV'])); m('authXSTe', f2(auth_beta['test']['XSR2_weighted']))
m('authEVTr', f2(auth_beta['train']['EV'])); m('authXSTr', f2(auth_beta['train']['XSR2_weighted']))
if our_beta:
    m('ganEVTe', f2(our_beta['test']['EV'])); m('ganXSTe', f2(our_beta['test']['XSR2_weighted']))
    m('ganEVTr', f2(our_beta['train']['EV'])); m('ganXSTr', f2(our_beta['train']['XSR2_weighted']))
else:
    for q in ('ganEVTe', 'ganXSTe', 'ganEVTr', 'ganXSTr'): m(q, '--')
tf_beta = opt('results/tf_ensemble_beta_stats.json')
if tf_beta:
    m('tfEVTe', f2(tf_beta['test']['EV'])); m('tfXSTe', f2(tf_beta['test']['XSR2_weighted']))
    m('tfEVTr', f2(tf_beta['train']['EV'])); m('tfXSTr', f2(tf_beta['train']['XSR2_weighted']))
else:
    for q in ('tfEVTe', 'tfXSTe', 'tfEVTr', 'tfXSTr'): m(q, '')
m('lsEVTe', f2(b['LS']['stats']['test']['EV'])); m('lsXSTe', f2(b['LS']['stats']['test']['XSR2_weighted']))
m('ridgeEVTe', f2(b['Ridge']['stats']['test']['EV'])); m('ridgeXSTe', f2(b['Ridge']['stats']['test']['XSR2_weighted']))
t = ext['test']
m('turnDollar', f2(t['dollar_turnover'])); m('turnLong', f2(t['turnover_long'])); m('turnShort', f2(t['turnover_short'])); m('srNetTen', f2(t['sr_net_10bp'])); m('srNetTwentyFive', f2(t['sr_net_25bp'])); m('srNetFifty', f2(t['sr_net_50bp']))
m('srEarly', f2(t['sr_1992_2004'])); m('srLate', f2(t['sr_2005_2016'])); m('srExGFC', f2(t['sr_excl_2008_2009']))
m('maxLoss', '%.1f' % (100 * t['max_1m_loss']))
pub = opt('results/published_sdf_check.json')
if pub:
    m('corrPortTe', f2(pub['test']['corr_port'])); m('corrPortTr', f2(pub['train']['corr_port']))
    m('corrOursTe', f2(pub['test']['corr_ours'])); m('corrOursTr', f2(pub['train']['corr_ours']))
else:
    for q in ('corrPortTe', 'corrPortTr', 'corrOursTe', 'corrOursTr'): m(q, '--')
open('report/numbers.tex', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
