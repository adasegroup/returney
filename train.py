import sys
sys.path.append('../data')

from data.OCON.dataset import get_ocon_train_val_loaders
from hydra.experimental import compose, initialize
from RNNSM.rnnsm import RNNSM
from RMTPP.rmtpp import RMTPP
from torch import optim
from collections import defaultdict
import torch
import numpy as np
from utils import calc_rmse, calc_auc, calc_recall


def validate(val_loader, model, prediction_start, prediction_end, device):
    all_preds = []
    all_targets = []

    is_rnnsm = isinstance(model, RNNSM)

    with torch.no_grad():
        for timestamps, cat_feats, num_feats, targets, lengths in val_loader:
            if is_rnnsm:
                o_j = model(cat_feats.to(device), num_feats.to(device), lengths)
                preds = model.predict(o_j, timestamps, lengths)
            else:
                o_j, deltas_pred, _ = model(cat_feats.to(device), num_feats.to(device), lengths)
                preds = model.predict(deltas_pred, o_j, timestamps, lengths)
            targets = targets.numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return calc_rmse(all_preds, all_targets), \
        calc_recall(all_preds, all_targets, prediction_end), \
        calc_auc(all_preds, all_targets, prediction_end)


def train(train_loader, val_loader, model, optimizer, train_cfg, global_config, device):
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)

    is_rnnsm = isinstance(model, RNNSM)

    for epoch in range(train_cfg.n_epochs):
        print(f'Epoch {epoch+1}/{train_cfg.n_epochs}....')
        losses = []
        model.train()

        for timestamps, cat_feats, num_feats, non_pad_mask, return_mask, lengths in train_loader:
            deltas = timestamps[:, 1:] - timestamps[:, :-1]
            if is_rnnsm:
                o_j = model(cat_feats.to(device), num_feats.to(device), lengths)
                loss = model.compute_loss(deltas, non_pad_mask, return_mask, o_j)
            else:
                o_j, deltas_pred, y_j = model(cat_feats.to(device), num_feats.to(device), lengths)
                loss = model.compute_loss(deltas_pred, deltas, non_pad_mask, o_j, y_j, cat_feats)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_metrics['loss'].append(np.mean(losses))
        print("Loss:", train_metrics['loss'][-1])

        model.eval()
        rmse, recall, auc = validate(val_loader,
            model,
            global_config.prediction_start,
            global_config.prediction_end,
            device)
        val_metrics['rmse'].append(rmse)
        val_metrics['recall'].append(recall)
        val_metrics['auc'].append(auc)
        print(f'Validation: '
              f'RMSE: {val_metrics["rmse"][-1]},\t'
              f'Recall: {val_metrics["recall"][-1]},\t'
              f'AUC: {val_metrics["auc"][-1]} ')
    torch.save(model.state_dict(), train_cfg.model_path)
    return train_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    initialize(config_path=".")
    config = compose(config_name="config.yaml")

    model = config.training.model
    assert(model in ['rnnsm', 'rmtpp'])

    model_class = RNNSM if model == 'rnnsm' else RMTPP
    model_config = config.rnnsm if model == 'rnnsm' else config.rmtpp

    train_loader, val_loader = get_ocon_train_val_loaders(
                                 cat_feat_name='event_type',
                                 num_feat_name='time_delta',
                                 model=model,
                                 global_config=config.globals,
                                 path='data/OCON/train.csv',
                                 batch_size=config.training.batch_size,
                                 max_seq_len=model_config.max_seq_len)

    model = model_class(model_config, config.globals).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    train(train_loader, val_loader, model, optimizer, config.training, config.globals, device)


if __name__ == '__main__':
    main()
