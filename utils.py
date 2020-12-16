from sklearn.metrics import roc_auc_score, recall_score
import numpy as np


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
