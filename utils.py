import numpy as np
from sklearn.metrics import roc_auc_score, recall_score


def calc_rmse(predicted, target):
    predicted = predicted[target != -1]
    target = target[target != -1]
    return np.sqrt(np.mean((predicted - target) ** 2))


def calc_recall(predicted, target, prediction_end):
    y_pred = predicted > prediction_end
    y_true = target == -1
    return recall_score(y_true, y_pred)


def calc_auc(predicted, target, prediction_end):
    y_pred = predicted
    y_true = target == -1

    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return 'Undefined'


def rnnsm_test_step(model, device, timestamps, cat_feats, num_feats, targets, lengths):
    o_j = model(cat_feats.to(device), num_feats.to(device), lengths)
    preds = model.predict(o_j, timestamps, lengths)
    return preds


def rmtpp_test_step(model, device, timestamps, cat_feats, num_feats, targets, lengths):
    o_j, _ = model(cat_feats.to(device), num_feats.to(device), lengths)
    preds = model.predict(o_j, timestamps, lengths)
    return preds


def grobformer_test_step(model, device, timestamps, cat_feats, num_feats, targets, lengths):
    o_j = model(cat_feats.to(device), timestamps.to(device), lengths)
    preds = model.predict(o_j, timestamps, lengths)
    return preds
