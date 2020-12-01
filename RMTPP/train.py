import sys
sys.path.append('../data')
from ATM.dataset import get_ATM_loader
import argparse
import os
from hydra.experimental import compose, initialize
from rmtpp import RMTPP
from torch import optim
from collections import defaultdict
import torch
import numpy as np

def train_rmtpp(data_loader, model, path, cfg, device):
    train_metrics = defaultdict(list)

    for epoch in range(cfg.training.n_epochs):
        print(f'Epoch {epoch+1}/{cfg.training.n_epochs}....')
        losses = []
        for times, events, lengths, _ in data_loader:
            time_deltas = torch.cat([torch.zeros(times.shape[0], 1, times.shape[2]),
                                     times[:, 1:] - times[:, :-1]],
                                     dim=1)
            o_t, y_t = model(events.to(device), time_deltas.to(device), lengths)
            padding_mask = torch.isclose(times, torch.Tensor([0])).to(device)
            ret_mask = ~padding_mask
            loss = model.compute_loss(time_deltas, padding_mask, o_t, y_t[0], events)
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        train_metrics['loss'].append(np.mean(losses))
        print("Loss:", train_metrics['loss'][-1])
    torch.save(model.state_dict(), path)
    return train_metrics

parser = argparse.ArgumentParser()

parser.add_argument('--print_result',
        action='store_const',
        const=True,
        default=False,
        help='Print the integration result')


parser.add_argument('--use_gpu',
        action='store_const',
        const=True,
        default=False,
        help='Using GPU')

parser.add_argument('-bs', '--batch_size',
        type=int,
        default=32,
        help='Batch size')

parser.add_argument('-e', '--n_epochs',
        type=int,
        default=32,
        help='Batch size')

parser.add_argument('-seq', '--max_sequence_length',
        type=int,
        default=500,
        help='Maximum length of sequences to train on')

parser.add_argument('-d', '--dataset',
        type=str,
        default='ATM',
        help='Dataset to use. Allowed values: ["ATM"]')

parser.add_argument('-p', '--model_path',
        type=str,
        default='./model.pth',
        help='Path where to save the model.')


args = parser.parse_args()

if args.use_gpu:
    device = 'cuda'
else:
    device = 'cpu'

if args.dataset == "ATM":
    data_loader = get_ATM_loader(path='../data/ATM/train_day.csv', 
                                 batch_size=args.batch_size,
                                 max_seq_len=args.max_sequence_length)
else:
    data_loader = get_ATM_loader(path='../data/ATM/train_day.csv', 
                                 batch_size=args.batch_size,
                                 max_seq_len=args.max_sequence_length)

initialize(config_path="..")
cfg = compose(config_name="config.yaml")

model = RMTPP(cfg.rmtpp).to(device)
opt = optim.Adam(model.parameters(), lr=cfg.training.lr)
train_rmtpp(data_loader, model, args.model_path, cfg, device)

