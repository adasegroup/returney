import sys

sys.path.append('../data')

from data.OCON.dataset import get_ocon_test_loader
from hydra.experimental import compose, initialize
from RNNSM.rnnsm import RNNSM
from RMTPP.rmtpp import RMTPP
from grobformer.grobformer import Grobformer
import torch
from utils import *


def test(val_loader, model, global_cfg, device):
    all_preds = []
    all_targets = []

    model2test_step = {RNNSM: rnnsm_test_step, RMTPP: rmtpp_test_step, \
                       Grobformer: grobformer_test_step}
    test_step = model2test_step[type(model)]

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            preds = test_step(model, device, *batch)
            targets = batch[-2]
            targets = targets.numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    print(f'Testing: '
          f'RMSE: {calc_rmse(all_preds, all_targets)},\t'
          f'Recall: {calc_recall(all_preds, all_targets, global_cfg.prediction_end)},\t'
          f'AUC: {calc_auc(all_preds, all_targets, global_cfg.prediction_end)}')


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
                                       path='data/OCON/test.csv',
                                       batch_size=cfg.training.batch_size,
                                       max_seq_len=model_cfg.max_seq_len)

    model = model_class(model_cfg, cfg.globals)
    model.load_state_dict(torch.load(cfg.testing.model_path))
    test(test_loader, model, cfg.globals, device)


if __name__ == '__main__':
    main()
