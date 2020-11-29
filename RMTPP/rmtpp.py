import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class RMTPP(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        input_size, hidden_size = model_cfg.input_size, model_cfg.hidden_size
        self.hidden = nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.n_num_feats = model_cfg.n_num_feats

        cat_sizes = model_cfg.cat_sizes
        emb_dims = model_cfg.emb_dims
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size + 1, emb_dim, padding_idx=0) for cat_size, emb_dim
                                         in zip(cat_sizes, emb_dims)])

        self.marker_outs = nn.ModuleList([nn.Linear(hidden_size, cat_size+1) for cat_size in cat_sizes])
        self.input_dropout = nn.Dropout(model_cfg.dropout)

        self.marker_weights = model_cfg.marker_weights
        self.output_past_influence = nn.Linear(hidden_size, 1)
        self.w = model_cfg.w
        self.time_scale = model_cfg.time_scale


    def forward(self, cat_feats, num_feats, lengths):
        x = torch.zeros((cat_feats.size(0), cat_feats.size(1), 0)).to(cat_feats.device)
        for i, emb in enumerate(self.embeddings):
            x = torch.cat([x, emb(cat_feats[:, :, i])], axis=-1)
        x = torch.cat([x, num_feats], axis=-1)
        x = self.input_dropout(x)
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        h_j, _ = self.hidden(x)
        h_j, lengths = pad_packed_sequence(h_j, batch_first=True)
        o_j = self.output_past_influence(h_j).squeeze()
        ys_j = []
        for out in self.marker_outs:
            ys_j.append(out(h_j))

        return o_j, ys_j


    def compute_loss(self, deltas, padding_mask, o_j, ys_j, ys_true):
        deltas_scaled = deltas * self.time_scale
        time_prediction_loss = (-(torch.exp(o_j) / self.w - \
                                  torch.exp(o_j + self.w * deltas_scaled) / self.w + \
                                  self.w * deltas_scaled + o_j) * ~padding_mask).sum()
        markers_loss = 0
        for i, (w, y_j, y_true) in enumerate(zip(self.marker_weights, ys_j, ys_true)):
            markers_loss += w * F.cross_entropy(y_j.flatten(0, -2), y_true.flatten(), ignore_index=0, reduction='sum')
        return (time_prediction_loss + markers_loss) / torch.sum(~padding_mask)


    def predict(self, o_j, t_j):
        """
        Predicts the timing of the next event
        """
        t_j_scaled = t_j * self.time_scale
        last_o_j = o_j[torch.arange(o_j.size(0)), lengths]
        deltas = torch.arange(0, 1000 * self.time_scale, self.time_scale).to(o_j.device)
        timesteps = t_j_scaled[:, None] + deltas[None, :]

        f_deltas = self._f_t(last_o_j, deltas, broadcast_deltas=True)
        tf_deltas = timesteps * f_deltas
        return t_j.cpu().numpy() + trapz(tf_deltas.cpu(), timesteps.cpu()) / self.time_scale


    def _f_t(self, last_o_j, deltas, broadcast_deltas=False):
        if broadcast_deltas:
            lambda_t = torch.exp(last_o_j[:, None] + self.w * deltas[None, :])
            f_t = torch.exp(torch.log(lambda_t) + torch.exp(last_o_j)[:, None] / self.w - lambda_t / self.w)
        else:
            lambda_t = torch.exp(last_o_j + self.w * deltas)
            f_t = torch.exp(torch.log(lambda_t) + torch.exp(last_o_j) / self.w - lambda_t / self.w)
        return f_t


    def _s_t(self, last_o_j, deltas, broadcast_deltas=False):
        if broadcast_deltas:
            out = torch.exp(torch.exp(last_o_j)[:, None] / self.w - \
                    torch.exp(last_o_j[:, None] + self.w * deltas[None, :]) / self.w)
        else:
            out = torch.exp(torch.exp(last_o_j) / self.w - \
                    torch.exp(last_o_j + self.w * deltas) / self.w)
        return out
