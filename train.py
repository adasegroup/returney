import sys
sys.path.append('../data')

from data.OCON.dataset import get_ocon_train_val_loaders
from hydra.experimental import compose, initialize
from RNNSM.rnnsm import RNNSM
from RMTPP.rmtpp import RMTPP
from grobformer.grobformer import Grobformer
from torch import optim
from collections import defaultdict
import torch
import numpy as np
from utils import *


def validate(val_loader, model, prediction_start, prediction_end, device):
    all_preds = []
    all_targets = []

    model2test_step = {RNNSM: rnnsm_test_step, RMTPP: rmtpp_test_step, \
                        Grobformer: grobformer_test_step}
    test_step = model2test_step[type(model)]

    with torch.no_grad():
        for batch in val_loader:
            preds = test_step(model, device, *batch)
            targets = batch[-2]
            targets = targets.numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return calc_rmse(all_preds, all_targets), \
        calc_recall(all_preds, all_targets, prediction_end), \
        calc_auc(all_preds, all_targets, prediction_end)


def rnnsm_train_step(model, device, timestamps, cat_feats, num_feats, non_pad_mask, return_mask, lengths):
    deltas = timestamps[:, 1:] - timestamps[:, :-1]
    o_j = model(cat_feats.to(device), num_feats.to(device), lengths)
    loss = model.compute_loss(deltas, non_pad_mask, return_mask, o_j)
    return loss


def rmtpp_train_step(model, device, timestamps, cat_feats, num_feats, non_pad_mask, return_mask, lengths):
    deltas = timestamps[:, 1:] - timestamps[:, :-1]
    o_j, y_j = model(cat_feats.to(device), num_feats.to(device), lengths)
    loss = model.compute_loss(deltas, non_pad_mask, o_j, y_j, cat_feats)
    return loss


def grobformer_train_step(model, device, timestamps, cat_feats, num_feats, non_pad_mask, return_mask, lengths):
    deltas = timestamps[:, 1:] - timestamps[:, :-1]
    o_j = model(cat_feats.to(device), timestamps.to(device), lengths)
    loss = model.compute_loss(deltas, non_pad_mask, return_mask, o_j)
    return loss


def train(train_loader, val_loader, model, optimizer, train_cfg, global_cfg, device):
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)

    model2train_step = {RNNSM: rnnsm_train_step, RMTPP: rmtpp_train_step, \
                        Grobformer: grobformer_train_step}
    train_step = model2train_step[type(model)]

    best_metric = 1e10 if train_cfg.validate_by == 'rmse' else -1

    for epoch in range(train_cfg.n_epochs):
        print(f'Epoch {epoch+1}/{train_cfg.n_epochs}....')
        losses = []
        model.train()

        for batch in train_loader:
            loss = train_step(model, device, *batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_metrics['loss'].append(np.mean(losses))
        print("Loss:", train_metrics['loss'][-1])

        model.eval()
        rmse, recall, auc = validate(val_loader,
            model,
            global_cfg.prediction_start,
            global_cfg.prediction_end,
            device)

        if train_cfg.validate_by == 'rmse' and rmse < best_metric:
            torch.save(model.state_dict(), train_cfg.model_path)
        elif train_cfg.validate_by == 'recall' and recall > best_metric:
            torch.save(model.state_dict(), train_cfg.model_path)
        elif auc > best_metric:
            torch.save(model.state_dict(), train_cfg.model_path)

        val_metrics['rmse'].append(rmse)
        val_metrics['recall'].append(recall)
        val_metrics['auc'].append(auc)
        print(f'Validation: '
              f'RMSE: {val_metrics["rmse"][-1]},\t'
              f'Recall: {val_metrics["recall"][-1]},\t'
              f'AUC: {val_metrics["auc"][-1]} ')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    initialize(config_path=".")
    cfg = compose(config_name="config.yaml")

    model = cfg.training.model
    assert model in ['rnnsm', 'rmtpp', 'grobformer'], 'Invalid model name'

    if model == 'rmtpp':
        assert(cfg.training.validate_by == 'rmse')
    else:
        assert(cfg.training.validate_by in ['rmse', 'recall', 'auc'])

    model2class = {'rnnsm': RNNSM, 'rmtpp': RMTPP, 'grobformer': Grobformer}
    model2cfg = {'rnnsm': cfg.rnnsm, 'rmtpp': cfg.rmtpp, 'grobformer': cfg.grobformer}

    model_class = model2class[model]
    model_cfg = model2cfg[model]

    train_loader, val_loader = get_ocon_train_val_loaders(
                                 cat_feat_name='event_type',
                                 num_feat_name='time_delta',
                                 model=model,
                                 global_cfg=cfg.globals,
                                 path='data/OCON/train.csv',
                                 batch_size=cfg.training.batch_size,
                                 max_seq_len=model_cfg.max_seq_len)

    model = model_class(model_cfg, cfg.globals).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    train(train_loader, val_loader, model, optimizer, cfg.training, cfg.globals, device)


if __name__ == '__main__':
    main()
