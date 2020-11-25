import torch

def arange2d(start, end, step):
    return torch.cat([torch.arange(s, e, step).unsqueeze(0) for s, e in zip(start, end)], axis=0).to(start.device)
