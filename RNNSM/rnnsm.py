import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.integrate import trapz
import numpy as np


class RNNSM(nn.Module):
    def __init__(self, model_cfg, ):
        super().__init__()
        input_size, hidden_size = model_cfg.input_size, model_cfg.hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        cat_sizes = model_cfg.cat_sizes
        emb_dims = model_cfg.emb_dims
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size+1, emb_dim, padding_idx=0) for cat_size, emb_dim in zip(cat_sizes, emb_dims)])

        total_emb_length = sum(emb_dims)
        self.input_dense = nn.Linear(total_emb_length + model_cfg.n_num_feats, input_size)
        self.input_dropout = nn.Dropout(model_cfg.dropout)
        self.output_dense = nn.Linear(hidden_size, 1)

        self.w = model_cfg.w
        self.time_scale = model_cfg.time_scale

    def forward(self, cat_feats, num_feats, lengths):
        x = torch.zeros((cat_feats.size(0), cat_feats.size(1), 0)).to(cat_feats.device)
        for i, emb in enumerate(self.embeddings):
            x = torch.cat([x, emb(cat_feats[:, :, i])], axis=-1)
        x = torch.cat([x, num_feats], axis=-1)
        x = self.input_dropout(torch.tanh(self.input_dense(x)))
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        h_t, _ = self.lstm(x)
        h_t, _ = pad_packed_sequence(h_t, batch_first=True)
        o_j = self.output_dense(h_t).squeeze()
        return o_j


    def compute_loss(self, deltas, padding_mask, ret_mask, o_j):
        o_j = o_j[..., None]
        deltas_scaled = deltas * self.time_scale
        p = o_j + self.w * deltas_scaled
        common_term = -(torch.exp(o_j) - torch.exp(p)) / self.w * ~padding_mask
        ret_term = -p * ret_mask
        return (common_term.sum() + ret_term.sum()) / torch.sum(~padding_mask)


    def predict_standard(self, o_j, t_j, lengths):
        last_o_j = o_j[torch.arange(o_j.size(0)), lengths]
        deltas = torch.arange(0, 1000 * self.time_scale, self.time_scale).to(o_j.device)

        s_deltas = self._s_t(last_o_j, deltas, broadcast_deltas=True)
        return t_j.cpu().numpy() + trapz(s_deltas.cpu(), deltas_scaled.cpu()) / self.time_scale


    def predict(self, o_j, t_j, lenghts, pred_start):
        last_o_j = o_j[torch.arange(o_j.size(0)), lengths]
        t_j_scaled = t_j * self.time_scale
        t_til_start = pred_start * self.time_scale - t_j_scaled
        s_t_s = self._s_t(last_o_j, t_til_start)

        zeros = torch.zeros(t_j.size()).to(t_j.device)
        deltas = torch.arange(0, 1000 * self.time_scale, self.time_scale).to(o_j.device)
        s_deltas = self._s_t(last_o_j, deltas)

        preds = np.zeros(o_j.size(0))
        for i, ith_deltas in enumerate(deltas):
            ith_t_s, ith_s_t_s = t_til_start[i], s_t_s[i]
            pred_delta = trapz(s_deltas[ith_deltas < ith_t_s].cpu(), ith_deltas[ith_deltas < ith_t_s].cpu()) + \
                           trapz(s_deltas[ith_deltas >= ith_t_s].cpu(), ith_deltas[ith_deltas >= ith_t_s].cpu()) / ith_s_t_s.item()
            preds[i] = t_j[i].cpu().numpy() + pred_delta / self.time_scale

        return preds


    def _s_t(self, last_o_j, deltas, broadcast_deltas=False):
        if broadcast_deltas:
            out = torch.exp(torch.exp(last_o_j)[:, None] / self.w - \
                    torch.exp(last_o_j[:, None] + self.w * deltas[None, :]) / self.w)
        else:
            out = torch.exp(torch.exp(last_o_j) / self.w - \
                    torch.exp(last_o_j + self.w * deltas) / self.w)
        return out


    def save_model(self, path):
        torch.save({'lstm': self.lstm.state_dict(),
                    'embeddings': self.embeddings.state_dict(),
                    'input_dense': self.input_dense.state_dict(),
                    'output_dense': self.output_dense.state_dict()
                    }, path)


    def load_model(self, path):
        params = torch.load(path)
        self.lstm.load_state_dict(params['lstm'])
        self.embeddings.load_state_dict(params['embeddings'])
        self.input_dense.load_state_dict(params['input_dense'])
        self.output_dense.load_state_dict(params['output_dense'])
