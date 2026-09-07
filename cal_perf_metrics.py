import os
import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix
)
from utils import load_model_dict
from models import init_model_dict
from train_test import prepare_trte_data, gen_trte_adj_mat, test_epoch

cuda = True if torch.cuda.is_available() else False


def _sens_spec_macro(y_true, y_pred, num_class):
    """Per-class sensitivity/specificity from the confusion matrix, macro-averaged.
    For num_class == 2 this reduces to the usual binary sensitivity/specificity."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_class)))
    sens_list, spec_list = [], []
    total = cm.sum()
    for c in range(num_class):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(sens_list)), float(np.mean(spec_list))


def load_trained_model(data_folder, model_folder, view_list, num_class):
    """Rebuild the model architecture and load trained weights from model_folder.
    Same steps cal_feat_imp uses before it starts perturbing features."""
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400, 400, 200]

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)

    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    model_dict = load_model_dict(model_folder, model_dict)

    return model_dict, data_trte_list, adj_te_list, trte_idx, labels_trte


def cal_test_metrics(data_folder, model_folder, view_list, num_class):
    """Run the trained model once on the held-out test split and compute the
    real classification metrics, in place of the np.random.uniform placeholders."""
    model_dict, data_trte_list, adj_te_list, trte_idx, labels_trte = load_trained_model(
        data_folder, model_folder, view_list, num_class
    )

    te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    y_true = labels_trte[trte_idx["te"]]
    y_pred = te_prob.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    sens, spec = _sens_spec_macro(y_true, y_pred, num_class)

    if num_class == 2:
        auc = roc_auc_score(y_true, te_prob[:, 1])
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
    else:
        auc = roc_auc_score(y_true, te_prob, multi_class='ovr', average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro')

    return {
        "auc": auc,
        "acc": acc,
        "bacc": bacc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "sensitivity": sens,
        "specificity": spec,
    }
