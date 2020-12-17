import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from scipy.integrate import trapz


class RMTPP(nn.Module):
    def __init__(self, cfg, global_cfg):
        super().__init__()
        self.lstm = nn.LSTM(cfg.input_size, cfg.lstm_hidden_size, batch_first=True)
        self.n_num_feats = cfg.n_num_feats

        cat_sizes = cfg.cat_sizes
        emb_dims = cfg.emb_dims
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size + 1, emb_dim, padding_idx=global_cfg.padding) for cat_size, emb_dim
                                         in zip(cat_sizes, emb_dims)])

        self.hidden = nn.Linear(cfg.lstm_hidden_size, cfg.hidden_size)
        self.marker_outs = nn.ModuleList([nn.Linear(cfg.hidden_size, cat_size + 1) for cat_size in cat_sizes])
        self.dropout = nn.Dropout(cfg.dropout)

        self.marker_weights = cfg.marker_weights
        self.time_loss_weight = cfg.time_loss_weight

        self.output_dense = nn.Linear(cfg.hidden_size, 1)
        self.time_pred_head = nn.Linear(cfg.hidden_size, 1, bias=False)

        if cfg.w_trainable:
            self.w = nn.Parameter(torch.FloatTensor([0.1]))
        else:
            self.w = cfg.w
        self.time_scale = cfg.time_scale
        self.integration_end = cfg.integration_end

    def forward(self, cat_feats, num_feats, lengths):
        x = torch.zeros(*cat_feats.size()[:2], 0).to(cat_feats.device)
        for i, emb in enumerate(self.embeddings):
            x = torch.cat([x, emb(cat_feats[:, :, i])], axis=-1)
        x = torch.cat([x, num_feats], axis=-1)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        h_j, _ = self.lstm(x)
        h_j, lengths = pad_packed_sequence(h_j, batch_first=True)
        h_j = self.dropout(torch.tanh(self.hidden(h_j)))

        deltas_pred = self.time_pred_head(h_j).squeeze()
        o_j = self.output_dense(h_j).squeeze()
        ys_j = []
        for out in self.marker_outs:
            ys_j.append(out(h_j))

        return o_j, deltas_pred, ys_j

    def compute_loss(self, deltas_pred, deltas, non_pad_mask, o_j, ys_j, ys_true):
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
        time_pred_loss = ((deltas_scaled - deltas_pred) ** 2).sum()
        return (neg_ll + markers_loss + time_pred_loss * self.time_loss_weight) \
            / torch.sum(non_pad_mask)

    def predict(self, deltas_pred, t_j, lengths):
        with torch.no_grad():
            batch_size = t_j.size(0)
            last_t_j = t_j[torch.arange(batch_size), lengths - 1]
            last_deltas_pred = deltas_pred[torch.arange(batch_size), lengths - 1]

            return last_t_j + last_deltas_pred / self.time_scale
