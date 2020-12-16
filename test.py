import sys
sys.path.append('../data')

from data.OCON.dataset import get_ocon_test_loader
from hydra.experimental import compose, initialize
from RNNSM.rnnsm import RNNSM
from RMTPP.rmtpp import RMTPP
from grobformer.grobformer import Grobformer
from torch import optim
from collections import defaultdict
import torch
import numpy as np
from utils import calc_auc, calc_rmse, calc_recall


def rnnsm_test_step(model, device, timestamps, cat_feats, num_feats, targets, lengths):
    o_j = model(cat_feats.to(device), num_feats.to(device), lengths)
    preds = model.predict(o_j, timestamps, lengths)
    return preds


def rmtpp_test_step(model, device, timestamps, cat_feats, num_feats, targets, lengths):
    o_j, deltas_pred, _ = model(cat_feats.to(device), num_feats.to(device), lengths)
    preds = model.predict(deltas_pred, timestamps, lengths)
    return preds


def grobformer_test_step(model, device, timestamps, cat_feats, num_feats, targets, lengths):
    o_j = model(cat_feats.to(device), timestamps.to(device), lengths)
    preds = model.predict(o_j, timestamps, lengths)
    return preds


def test(val_loader, model, prediction_start, prediction_end, device):
    all_preds = []
    all_targets = []

    model2test_step = {RNNSM: rnnsm_test_step, RMTPP: rmtpp_test_step, \
                        Grobformer: grobformer_test_step}
    test_step = model2test_step[type(model)]

    model.eval()
    with torch.no_grad():
        for timestamps, cat_feats, num_feats, targets, lengths in val_loader:
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    initialize(config_path=".")
    cfg = compose(config_name="config.yaml")

    model = cfg.testing.model
    assert model in ['rnnsm', 'rmtpp', 'grobformer'], 'Invalid model name'

    model2class = {'rnnsm': RNNSM, 'rmtpp': RMTPP, 'grobformer': Grobformer}
    model2cfg = {'rnnsm': cfg.rnnsm, 'rmtpp': cfg.rmtpp, 'grobformer': cfg.grobformer}

    model_class = model2class[model]
    model_cfg = model2cfg[model]

    test_loader = get_ocon_test_loader(cat_feat_name='event_type',
                                       num_feat_name='time_delta',
                                       global_cfg=cfg.globals,
                                       path='data/OCON/train.csv',
                                       batch_size=cfg.training.batch_size,
                                       max_seq_len=model_cfg.max_seq_len)

    model = model_class(model_cfg, cfg.globals)
    model.load_state_dict(cfg.testing.model_path)
    test(test_loader, model, prediction_start, prediction_end, device)


if __name__ == '__main__':
    main()
