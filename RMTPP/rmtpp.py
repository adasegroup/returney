import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RMTPP(nn.Module):
    def __init__(self, model_cfg):
        super(RMTPP, self).__init__()
        input_size, hidden_size, num_layers = model_cfg.input_size, model_cfg.hidden_size, model_cfg.num_layers
        self.hidden = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', batch_first=True)
        self.n_num_feats = model_cfg.n_num_feats
        
        
        cat_sizes = model_cfg.cat_sizes
        emb_dims = model_cfg.emb_dims
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size + 1, emb_dim, padding_idx=0) for cat_size, emb_dim
                                         in zip(cat_sizes, emb_dims)])
        
        self.marker_outs = nn.ModuleList([nn.Linear(hidden_size, cat_sizes) for cat_size in cat_sizes])
        self.output_past_influence = nn.Linear(hidden_size, 1)
        self.w = model_cfg.w
        self.b = model_cfg.b
        self.time_scale = model_cfg.time_scale
      
    
    def forward(self, cat_feats, num_feats, lengths):
        x = torch.zeros((cat_feats.size(0), cat_feats.size(1), 0)).to(cat_feats.device)
        for i, emb in enumerate(self.embeddings):
            x = torch.cat([x, emb(cat_feats[:, :, i])], axis=-1)
        x = torch.cat([x, num_feats], axis=-1)
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True)
        h_j = self.hidden(x)
        h_j = pad_packed_sequence(h_j)
        o_j = self.output_past_influence(h_j).squeeze()
        ys_j = []
        for layer in self.marker_outs:
            ys_j.append(layer(h_j))
        return o_j, ys_j
        
        
    def compute_loss(self, deltas, o_j, ys_j):
        total_loss = torch.log(self._f_t(o_j, deltas))
        for y_j in ys_j:
            total_loss += torch.log(torch.sum(torch.exp(y_j) / torch.sum(torch.exp(y_j))))
        return total_loss
    
    
    def predict(self, o_j, t_j):
        """
        Predicts the timing of the next event
        """
        last_o_j = o_j[torch.arange(o_j.size(0)), lengths] # why not o_j[:, lengths]?
        timesteps = arange2d(t_j, t_j + 1000 * self.time_scale, self.time_scale)
        deltas = timesteps - t_j[:, None]

        f_deltas = self._f_t(o_j, deltas)
        return trapz(f_deltas.cpu(), timesteps.cpu())


    def _f_t(self, o_j, deltas):
        lambda_t = torch.exp(o_j + self.w*deltas + self.b)
        f_t = torch.exp(lambda_t + 1/self.w * torch.exp(o_j + self.b) - 1/self.w * lambda_t)
        return f_t.to(o_j.device)