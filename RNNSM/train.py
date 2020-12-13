import sys
sys.path.append('../data')
from ATM.dataset import get_ATM_train_val_loaders
import argparse
import os
from hydra.experimental import compose, initialize
from rnnsm import RNNSM
from torch import optim
from collections import defaultdict
import torch
import numpy as np

from sklearn.metrics import roc_auc_score, recall_score

activity_start = 16000
prediction_start = 16400
prediction_end = 16500


def calc_rmse(predicted, target):
    print(predicted)
    predicted = predicted[target != -1]
    target = target[target != -1]
    return np.sqrt(np.mean((predicted - target) ** 2))


def calc_recall(predicted, target, prediction_end):
    y_pred = predicted > prediction_end
    y_true = target == -1
    return recall_score(y_true, y_pred)


def calc_auc(predicted, target, prediction_end):
    y_pred = predicted > prediction_end
    y_true = target == -1
    return roc_auc_score(y_true, y_pred)


def validate(val_loader, model, prediction_start, prediction_end, device):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for timestamps, cat_feats, num_feats, targets, lengths in val_loader:
            o_j = model(cat_feats.to(device), num_feats.to(device), lengths)
            t_next = model.predict(o_j, timestamps, lengths, prediction_start)
            targets = targets.numpy()

            all_preds.extend(t_next.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return calc_rmse(all_preds, all_targets), \
        calc_recall(all_preds, all_targets, prediction_end), \
        calc_auc(all_preds, all_targets, prediction_end)


def train_rnnsm(train_loader, val_loader, model, optimizer, path, cfg, device):
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)

    for epoch in range(cfg.training.n_epochs):
        print(f'Epoch {epoch+1}/{cfg.training.n_epochs}....')
        losses = []
        model.train()

        for timestamps, cat_feats, num_feats, padding_mask, return_mask, lengths in train_loader:
            time_deltas = timestamps[:, 1:] - timestamps[:, :-1]
            o_j = model(cat_feats.to(device), num_feats.to(device), lengths)
            loss = model.compute_loss(time_deltas, padding_mask, return_mask, o_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_metrics['loss'].append(np.mean(losses))
        print("Loss:", train_metrics['loss'][-1])

        model.eval()
        rmse, precision, recall = validate(val_loader,
            model,
            prediction_start,
            prediction_end,
            device)
        val_metrics['rmse'].append(rmse)
        val_metrics['precision'].append(precision)
        val_metrics['recall'].append(recall)
        print(f'Validation: '
              f'RMSE: {val_metrics["rmse"][-1]},'
              f'Precision: {val_metrics["precision"][-1]},'
              f'Recall: {val_metrics["precision"][-1]} ')
    torch.save(model.state_dict(), path)
    return train_metrics


def main():
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
            default=100,
            help='Batch size')

    parser.add_argument('-seq', '--max_sequence_length',
            type=int,
            default=100,
            help='Maximum length of sequences to train on')

    parser.add_argument('-d', '--dataset',
            type=str,
            default='ATM',
            help='Dataset to use. Allowed values: ["ATM"]')

    parser.add_argument('-p', '--model_path',
            type=str,
            default='./rnnsm_model.pth',
            help='Path where to save the model.')


    args = parser.parse_args()

    if args.use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    train_loader, val_loader = get_ATM_train_val_loaders(model='RNNSM',
                                 activity_start=activity_start,
                                 prediction_start=prediction_start,
                                 prediction_end=prediction_end,
                                 path='../data/ATM/train_day.csv',
                                 batch_size=args.batch_size,
                                 max_seq_len=args.max_sequence_length)

    initialize(config_path="..")
    cfg = compose(config_name="config.yaml")

    model = RNNSM(cfg.rnnsm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    train_rnnsm(train_loader, val_loader, model, optimizer, args.model_path, cfg, device)


if __name__ == '__main__':
    main()
