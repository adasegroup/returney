import sys
sys.path.append('../data')

from data.OCON.dataset import get_ocon_test_loader
from hydra.experimental import compose, initialize
from RNNSM.rnnsm import RNNSM
from RMTPP.rmtpp import RMTPP
from torch import optim
from collections import defaultdict
import torch
import numpy as np
from utils import calc_auc, calc_rmse, calc_recall


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
    config = compose(config_name="config.yaml")

    model = config.testing.model
    assert (model in ['rnnsm', 'rmtpp'])

    model_class = RNNSM if model == 'rnnsm' else RMTPP
    max_seq_len = config.rnnsm.max_seq_len if model == 'rnnsm' else \
        config.rmtpp.max_seq_len
    test_loader = get_ocon_test_loader(cat_feat_name='event_type',
                                       num_feat_name='time_delta',
                                       global_config=config.globals,
                                       path='data/OCON/train.csv',
                                       batch_size=config.training.batch_size,
                                       max_seq_len=max_seq_len)

    model = model_class(model_config, config.globals).load_model(config.testing.model_path)
    validate(test_loader, model, prediction_start, prediction_end, device)


if __name__ == '__main__':
    main()
