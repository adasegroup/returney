import sys

sys.path.append('../data')

from data.OCON.dataset import get_ocon_test_loader
import argparse
import os
from hydra.experimental import compose, initialize
from RNNSM.rnnsm import RNNSM
from RMTPP.rmtpp import RMTPP
from torch import optim
from collections import defaultdict
import torch
import numpy as np

from sklearn.metrics import roc_auc_score, recall_score

activity_start = 16000
prediction_start = 16400
prediction_end = 16500


def calc_rmse(predicted, target):
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

    is_rnnsm = isinstance(model, RNNSM)

    with torch.no_grad():
        for timestamps, cat_feats, num_feats, targets, lengths in val_loader:
            if is_rnnsm:
                o_j = model(cat_feats.to(device), num_feats.to(device), lengths)
                preds = model.predict(o_j, timestamps, lengths, prediction_start)
            else:
                o_j, _ = model(cat_feats.to(device), num_feats.to(device), lengths)
                preds = model.predict(o_j, timestamps, lengths)
            targets = targets.numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return calc_rmse(all_preds, all_targets), \
           calc_recall(all_preds, all_targets, prediction_end), \
           calc_auc(all_preds, all_targets, prediction_end)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    initialize(config_path=".")
    cfg = compose(config_name="config.yaml")

    model = cfg.testing.model
    assert (model in ['rnnsm', 'rmtpp'])

    model_class = RNNSM if model == 'rnnsm' else RMTPP
    max_seq_len = cfg.rnnsm.max_seq_len if model == 'rnnsm' else \
        cfg.rmtpp.max_seq_len
    test_loader = get_ocon_test_loader(cat_feat_name='event_type',
                                       num_feat_name='time_delta',
                                       global_config=cfg.globals,
                                       path='data/OCON/train.csv',
                                       batch_size=cfg.training.batch_size,
                                       max_seq_len=max_seq_len)

    model = model_class.load_model(cfg.testing.model_path)
    validate(test_loader, model, prediction_start, prediction_end, device)


if __name__ == '__main__':
    main()
