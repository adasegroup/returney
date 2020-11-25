import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.integrate import trapz

class GrobLSTM(nn.Module):
    def __init__(self, model_cfg):
        super(GrobLSTM, self).__init__()
        input_size, hidden_size, num_layers = model_cfg.input_size, model_cfg.hidden_size, model_cfg.num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.2, batch_first=True)
        cat_sizes = model_cfg.cat_sizes
        emb_dims = model_cfg.emb_dims
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, emb_dim, padding_idx=0) for cat_size, emb_dim in zip(cat_sizes, emb_dims)])

        total_emb_length = sum(emb_dims)
        self.input_dense = nn.Linear(total_emb_length + model_cfg.n_num_feats, input_size)
        self.input_dropout = nn.Dropout(0.2)
        self.output_dense = nn.Linear(hidden_size, 1)

        self.w = model_cfg.w
        self.time_scale = model_cfg.time_scale
        self.pred_start = pred_start * self.time_scale

    def forward(self, cat_feats, num_feats, lengths):
        x = torch.zeros((cat_feats.size(0), cat_feats.size(1), 0)).to(cat_feats.device)
        for i, emb in enumerate(self.embeddings):
            x = torch.cat([x, emb(cat_feats[:, :, i])], axis=-1)
        x = torch.cat([x, num_feats], axis=-1)
        x = self.input_dropout(torch.tanh(self.input_dense(x)))
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True)
        h_t, _ = self.lstm(x)
        h_t, _ = pad_packed_sequence(h_t)
        o_t = self.output_dense(h_t).squeeze()
        return o_t

    def compute_loss(self, deltas, padding_mask, ret_mask, o_t):
        common_term = -(torch.exp(o_t) / self.w - torch.exp(outputs + self.w * deltas) / self.w) * ~padding_mask
        ret_term = -(self.w * deltas + o_t) * ret_mask
        return common_term + ret_term

    def predict_standard(self, o_t, t_j, lengths):
        last_o_t = o_t[torch.arange(o_t.size(0)), lengths]
        timesteps = arange2d(t_j, t_j + 1000 * self.time_scale, self.time_scale)
        deltas = timesteps - t_j[:, None]

        s_deltas = self._s_t(last_o_t, deltas)
        return trapz(s_deltas.cpu(), timesteps.cpu())

    def predict(self, o_t, t_j, lenghts):
        last_o_t = o_t[torch.arange(o_t.size(0)), lengths]
        t_til_start = self.pred_start - t_j
        s_t_s = self._s_t(last_o_t, t_til_start)

        timesteps = arange2d(t_j, t_j + 1000 * self.time_scale, self.time_scale)
        deltas = timesteps - t_j[:, None]
        s_deltas = self._s_t(last_o_t, deltas)

        batch_size = timesteps.size(0)
        preds = torch.zeros(batch_size)
        for i, (ith_ts, ith_deltas) in enumerate(zip(timesteps, deltas)):
            ith_t_s, ith_s_t_s = t_til_start[i], s_t_s[i]
            preds[i] = trapz(ith_deltas[ith_ts < ith_t_s].cpu(), ith_ts[ith_ts < ith_t_s].cpu()) + \
                           trapz(ith_deltas[ith_ts >= ith_t_s].cpu(), ith_ts[ith_ts >= ith_t_s].cpu())

        return preds

    def _s_t(self, last_o_t, deltas):
        if len(deltas.size()) == 2:
            out = torch.exp(torch.exp(last_o_t)[:, None] / self.w - torch.exp(last_o_t[:, None] + self.w * deltas) / self.w)
        else:
            out = torch.exp(torch.exp(last_o_t) / self.w - torch.exp(last_o_t + self.w * deltas) / self.w)
        return out.to(last_o_t.device)
