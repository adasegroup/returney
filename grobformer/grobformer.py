import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from RNNSM.rnnsm import RNNSM
from grobformer.transformer import Transformer


class Grobformer(RNNSM):
    def __init__(self, cfg, global_cfg):
        nn.Module.__init__(self)

        d_model = cfg.model_dim
        hidden_dim = cfg.hidden_dim
        n_head = cfg.n_head
        n_layers = cfg.n_layers
        d_k = cfg.d_k
        d_v = cfg.d_v
        dropout = cfg.dropout

        num_types = cfg.cat_size
        self.transformer = Transformer(global_cfg, num_types, d_model, hidden_dim,
                                       n_layers, n_head, d_k, d_v, dropout)

        self.output_dense = nn.Linear(d_model, 1, bias=False)

        if cfg.w_trainable:
            self.w = nn.Parameter(torch.FloatTensor([0.1]))
        else:
            self.w = cfg.w

        self.time_scale = cfg.time_scale
        self.prediction_start = global_cfg.prediction_start
        self.integration_end = cfg.integration_end

    def forward(self, cat_feats, times, lengths):
        # times = times * self.time_scale
        # if training
        if times.size(1) > cat_feats.size(1):
            times = times[:, :-1]
        non_pad_mask = pad_sequence(
            [torch.ones(l).ne(0) for l in lengths], batch_first=True).to(cat_feats.device)
        hidden_states = self.transformer(cat_feats.squeeze(), times.squeeze(), non_pad_mask)
        o_j = self.output_dense(hidden_states).squeeze()
        return o_j
