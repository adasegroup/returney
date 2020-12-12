import sys

sys.path.append('../data')
from ATM.dataset import get_ATM_train_val_loaders
import argparse
import os
from hydra.experimental import compose, initialize
from rmtpp import RMTPP
from torch import optim
from collections import defaultdict
import torch
import numpy as np


def calc_rmse(predicted, target, prediction_end):
    predicted[np.where(predicted > prediction_end)] = -1
    return np.sqrt(np.mean((predicted - target) ** 2))


def calc_precision(predicted, target, prediction_end):
    return np.where(target[np.where(predicted < prediction_end)] > -1) / np.where(predicted < prediction_end)[0].shape[
        0]


def calc_recall(predicted, target, prediction_end):
    return np.where(target[np.where(predicted < prediction_end)] > -1) / np.where(target > -1)[0].shape[0]


def validate(val_data_loader, model, prediction_start, prediction_end):
    rmse = []
    precision = []
    recall = []
    for times, time_deltas, targets, events, lengths in val_data_loader:
        o_t, y_t = model(events, time_deltas, lengths)
        t_next = model.predict(o_t, torch.squeeze(times), lengths)
        targets = torch.squeeze(targets).detach().numpy()
        rmse.append(calc_rmse(t_next, targets, prediction_end))
    return np.mean(rmse), np.mean(precision), np.mean(recall)


def train_rmtpp(train_data_loader, val_data_loader, model, path, cfg, device):
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)

    for epoch in range(cfg.training.n_epochs):
        print(f'Epoch {epoch + 1}/{cfg.training.n_epochs}....')
        losses = []
        for times, events, lengths, _ in train_data_loader:
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

        rmse, precision, recall = validate(val_data_loader, model, prediction_start, prediction_end)
        val_metrics['rmse'].append(rmse)
        val_metrics['precision'].append(rmse)
        val_metrics['recall'].append(rmse)
        print(f'Validation: '
              f'RMSE: {val_metrics["rmse"][-1]},'
              f'Precision: {val_metrics["precision"][-1]},'
              f'Recall: {val_metrics["precision"][-1]} ')

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
activity_start = 16000
prediction_start = 16400
prediction_end = 16500
if args.dataset == "ATM":
    data_loader = get_ATM_train_val_loaders(activity_start=activity_start,
                                            prediction_start=prediction_start,
                                            prediction_end=prediction_end,
                                            path='../data/ATM/train_day.csv',
                                            batch_size=args.batch_size,
                                            max_seq_len=args.max_sequence_length)
else:
    data_loader = get_ATM_train_val_loaders(activity_start=activity_start,
                                            prediction_start=prediction_start,
                                            prediction_end=prediction_end,
                                            path='../data/ATM/train_day.csv',
                                            batch_size=args.batch_size,
                                            max_seq_len=args.max_sequence_length)

initialize(config_path="..")
cfg = compose(config_name="config.yaml")

model = RMTPP(cfg.rmtpp).to(device)
opt = optim.Adam(model.parameters(), lr=cfg.training.lr)
train_rmtpp(data_loader, model, args.model_path, cfg, device)
