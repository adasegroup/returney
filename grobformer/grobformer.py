import torch
from torch import nn
from scipy.integrate import trapz
import numpy as np
from grobformer.transformer import Transformer
from RNNSM.rnnsm import RNNSM


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

        num_types = cfg.cat_sizes[0]
        self.transformer = Transformer(num_types, d_model, hidden_dim, n_layers,
                                       n_head, d_k, d_v, dropout)

        self.output_dense = nn.Linear(d_model, 1, bias=False)

        if cfg.w_trainable:
            self.w = nn.Parameter(torch.FloatTensor([0.1]))
        else:
            self.w = cfg.w

        self.time_scale = cfg.time_scale
        self.prediction_start = cfg.prediction_start
        self.integration_end = cfg.integration_end

    def forward(self, cat_feats, times, lengths):
        times = times * self.time_scale
        non_pad_mask = pad_sequence(
            [torch.ones(l).ne(0) for l in lengths], batch_first=True).to(cat_feats.device)
        hidden_states = self.transformer(cat_feats, times, non_pad_mask)
        o_j = self.output_dense(hidden_states).squeeze()
        return o_j
