"""
Load the authors' 9 pretrained TF-1.12 checkpoints (sample_checkpoints.zip) into the PyTorch SDFNet and
recompute the ensemble Sharpe ratios.  If the port is faithful, this must reproduce the numbers hard-coded in
the authors' notebook / Table I:  SR train 2.68, valid 1.43, test 0.75.
"""
import os, json
import numpy as np, torch
import tensorflow as tf
from gan_sdf import Split, SDFNet, evaluate, normalized_factor, sharpe

CK = 'data/sample_checkpoints'
S = {k: Split(k, device='cpu') for k in ('train',)}
S['valid'] = Split('valid', S['train'].macro_mean, S['train'].macro_std)
S['test'] = Split('test', S['train'].macro_mean, S['train'].macro_std)


def tf_lstm_to_torch(K, B, D, H):
    """TF LSTMCell kernel [(D+H),4H] gates (i,j,f,o), forget_bias=1 added in-cell -> torch (i,f,g,o)."""
    Ki, Kh = K[:D], K[D:]
    i, j, f, o = [slice(k * H, (k + 1) * H) for k in range(4)]
    order = lambda M: np.concatenate([M[:, i], M[:, f], M[:, j], M[:, o]], 1)
    b = B.copy(); b[f] += 1.0
    return (torch.tensor(order(Ki).T.copy()), torch.tensor(order(Kh).T.copy()),
            torch.tensor(np.concatenate([b[i], b[f], b[j], b[o]])), torch.zeros(4 * H))


def load_trial(k):
    rd = tf.train.load_checkpoint(f'{CK}/Task_1_Trial_{k}/sharpe/model-best')
    g = lambda n: rd.get_tensor(n)
    net = SDFNet()
    wih, whh, bih, bhh = tf_lstm_to_torch(g('Model_Layer/RNN_Layer/rnn/lstm_cell/kernel'),
                                          g('Model_Layer/RNN_Layer/rnn/lstm_cell/bias'), 178, 4)
    sd = net.state_dict()
    sd['lstm.weight_ih_l0'], sd['lstm.weight_hh_l0'], sd['lstm.bias_ih_l0'], sd['lstm.bias_hh_l0'] = wih, whh, bih, bhh
    for tl, pl in [('dense_layer_0', 'ffn.0'), ('dense_layer_1', 'ffn.3'), ('last_dense_layer', 'ffn.6')]:
        sd[f'{pl}.weight'] = torch.tensor(g(f'Model_Layer/NN_Layer/{tl}/dense/kernel').T.copy())
        sd[f'{pl}.bias'] = torch.tensor(g(f'Model_Layer/NN_Layer/{tl}/dense/bias'))
    net.load_state_dict(sd)
    return net


os.makedirs('results', exist_ok=True)
W = {k: [] for k in S}; states0 = None; per_trial = []
for k in range(9):
    ev = evaluate(load_trial(k), S)
    for s in S: W[s].append(ev[s]['w'])
    if k == 0: states0 = {s: ev[s]['states'] for s in S}
    per_trial.append({s: sharpe(normalized_factor(ev[s]['w'], S[s])[0]) for s in S})
    print(f'trial {k}: SR train/valid/test = %.2f / %.2f / %.2f' % tuple(per_trial[-1].values()))

out = {}
for s in S:
    w_bar = np.mean(W[s], 0)
    F, wn = normalized_factor(w_bar, S[s])
    out[s] = sharpe(F)
    np.savez(f'results/authors_ensemble_{s}.npz', w=w_bar, w_norm=wn, F=F, states=states0[s], dates=S[s].dates)
print('AUTHORS 9-MODEL ENSEMBLE via PyTorch port:  SR train/valid/test = %.2f / %.2f / %.2f   (paper Table I: 2.68 / 1.43 / 0.75)'
      % (out['train'], out['valid'], out['test']))
json.dump(dict(ensemble=out, per_trial=per_trial), open('results/authors_checkpoints_sr.json', 'w'), indent=1)
