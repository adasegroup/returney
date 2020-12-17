import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from scipy.integrate import trapz


class RMTPP(nn.Module):
    def __init__(self, cfg, global_cfg):
        super().__init__()
        if cfg.use_lstm:
            self.rnn = nn.LSTM(cfg.input_size, cfg.rnn_hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(cfg.input_size, cfg.rnn_hidden_size, batch_first=True, nonlinearity='relu')
        self.n_num_feats = cfg.n_num_feats

        cat_sizes = cfg.cat_sizes
        emb_dims = cfg.emb_dims
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size + 1, emb_dim, padding_idx=global_cfg.padding) for cat_size, emb_dim
                                         in zip(cat_sizes, emb_dims)])

        self.hidden = nn.Linear(cfg.rnn_hidden_size, cfg.hidden_size)
        self.marker_outs = nn.ModuleList([nn.Linear(cfg.hidden_size, cat_size + 1) for cat_size in cat_sizes])
        self.dropout = nn.Dropout(cfg.dropout)

        self.marker_weights = cfg.marker_weights

        self.output_dense = nn.Linear(cfg.hidden_size, 1)

        if cfg.w_trainable:
            self.w = nn.Parameter(torch.FloatTensor([0.1]))
        else:
            self.w = cfg.w
        self.time_scale = cfg.time_scale
        self.integration_end = cfg.integration_end
        self.prediction_start = global_cfg.prediction_start

    def forward(self, cat_feats, num_feats, lengths):
        x = torch.zeros(*cat_feats.size()[:2], 0).to(cat_feats.device)
        for i, emb in enumerate(self.embeddings):
            x = torch.cat([x, emb(cat_feats[:, :, i])], axis=-1)
        x = torch.cat([x, num_feats], axis=-1)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        h_j, _ = self.rnn(x)
        h_j, lengths = pad_packed_sequence(h_j, batch_first=True)
        h_j = self.dropout(torch.tanh(self.hidden(h_j)))

        o_j = self.output_dense(h_j).squeeze()
        ys_j = []
        for out in self.marker_outs:
            ys_j.append(out(h_j))

        return o_j, ys_j

    def predict(self, o_j, t_j, lengths):
        with torch.no_grad():
            batch_size = o_j.size(0)
            last_o_j = o_j[torch.arange(batch_size), lengths - 1]
            last_t_j = t_j[torch.arange(batch_size), lengths - 1]

            t_til_start = (self.prediction_start - last_t_j) * self.time_scale
            s_t_s = self._s_t(last_o_j, t_til_start)

            preds = np.zeros(batch_size)
            for i in range(batch_size):
                ith_t_s, ith_s_t_s = t_til_start[i], s_t_s[i]
                deltas = torch.arange(0,
                    (self.integration_end - last_t_j[i]) * self.time_scale,
                    self.time_scale).to(o_j.device)
                ith_s_deltas = self._s_t(last_o_j[i], deltas[None, :])
                pred_delta = trapz(ith_s_deltas[deltas < ith_t_s].cpu(),
                    deltas[deltas < ith_t_s].cpu())
                pred_delta += trapz(ith_s_deltas[deltas >= ith_t_s].cpu(),
                    deltas[deltas >= ith_t_s].cpu()) / ith_s_t_s.item()
                preds[i] = last_t_j[i].cpu().numpy() + pred_delta / self.time_scale
        return preds

    def compute_loss(self, deltas, non_pad_mask, o_j, ys_j, ys_true):
        deltas_scaled = deltas * self.time_scale
        p = o_j + self.w * deltas_scaled
        ll = (p + (torch.exp(o_j) - torch.exp(p)) / self.w) * non_pad_mask
        neg_ll = -ll.sum()
        markers_loss = 0
        for i, (w, y_j) in enumerate(zip(self.marker_weights, ys_j)):
            y_true = ys_true[:, :, i]
            n_classes = y_j.size(-1)
            markers_loss += w * F.cross_entropy(y_j.view(-1, n_classes),
                y_true.view(-1),
                ignore_index=0,
                reduction='sum')

        return (neg_ll + markers_loss) / torch.sum(non_pad_mask)

    def _s_t(self, last_o_j, deltas):
        out = torch.exp(torch.exp(last_o_j) / self.w - \
            torch.exp(last_o_j + self.w * deltas) / self.w)
        return out.squeeze()
