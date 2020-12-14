import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from scipy.integrate import trapz


class RMTPP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(cfg.input_size, cfg.lstm_hidden_size, batch_first=True)
        self.n_num_feats = cfg.n_num_feats

        cat_sizes = cfg.cat_sizes
        emb_dims = cfg.emb_dims
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size + 1, emb_dim, padding_idx=0) for cat_size, emb_dim
                                         in zip(cat_sizes, emb_dims)])

        self.hidden = nn.Linear(cfg.lstm_hidden_size, cfg.hidden_size)
        self.marker_outs = nn.ModuleList([nn.Linear(cfg.hidden_size, cat_size + 1) for cat_size in cat_sizes])
        self.dropout = nn.Dropout(cfg.dropout)

        self.marker_weights = cfg.marker_weights
        self.output_dense = nn.Linear(cfg.hidden_size, 1)
        self.w = cfg.w
        self.time_scale = cfg.time_scale

        self.integration_steps = cfg.integration_steps

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
        o_j = self.output_dense(h_j).squeeze()
        ys_j = []
        for out in self.marker_outs:
            ys_j.append(out(h_j))

        return o_j, ys_j


    def compute_loss(self, deltas, padding_mask, o_j, ys_j, ys_true):
        deltas_scaled = deltas * self.time_scale
        p = o_j + self.w * deltas_scaled
        time_prediction_loss = (p + (torch.exp(o_j) - torch.exp(p)) / self.w) * ~padding_mask
        time_prediction_loss = -time_prediction_loss.sum()
        markers_loss = 0
        for i, (w, y_j) in enumerate(zip(self.marker_weights, ys_j)):
            y_true = ys_true[:, :, i]
            n_classes = y_j.size(-1)
            markers_loss += w * F.cross_entropy(y_j.view(-1, n_classes),
                y_true.view(-1),
                ignore_index=0,
                reduction='sum')
        return (time_prediction_loss + markers_loss) / torch.sum(~padding_mask)


    def predict(self, o_j, t_j, lengths):
        """
        Predicts the timing of the next event
        """
        with torch.no_grad():
            batch_size = t_j.size(0)
            last_t_j = t_j[torch.arange(batch_size), lengths - 1]
            last_o_j = o_j[torch.arange(batch_size), lengths - 1]

            deltas = torch.arange(0, self.integration_steps * self.time_scale, self.time_scale).to(o_j.device)

            f_deltas = self._f_t(last_o_j, deltas, broadcast_deltas=True)

            tf_deltas = deltas[None, :] * f_deltas
            print(trapz(tf_deltas.cpu(), deltas[None, :].cpu()) / self.time_scale)
            result = last_t_j.cpu().numpy() + trapz(tf_deltas.cpu(), deltas[None, :].cpu()) / self.time_scale
        return result


    def _f_t(self, last_o_j, deltas, broadcast_deltas=False):
        if broadcast_deltas:
            lambda_t = torch.exp(last_o_j[:, None] + self.w * deltas[None, :])
            f_t = torch.exp(torch.log(lambda_t) + torch.exp(last_o_j)[:, None] / self.w - lambda_t / self.w)
        else:
            lambda_t = torch.exp(last_o_j + self.w * deltas)
            f_t = torch.exp(torch.log(lambda_t) + torch.exp(last_o_j) / self.w - lambda_t / self.w)
        return f_t


    def save_model(self, path):
        torch.save({'lstm': self.lstm.state_dict(),
                    'embeddings': self.embeddings.state_dict(),
                    'output_dense': self.output_dense.state_dict(),
                    'hidden': self.hidden.state_dict()
                    }, path)


    def load_model(self, path):
        params = torch.load(path)
        self.lstm.load_state_dict(params['lstm'])
        self.embeddings.load_state_dict(params['embeddings'])
        self.output_dense.load_state_dict(params['output_dense'])
        self.hidden.load_state_dict(params['hidden'])
