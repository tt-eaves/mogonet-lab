"""
Additive training-history logger for MOGONET.

This module does NOT modify mogonet/train_test.py, mogonet/models.py, or
mogonet/utils.py. It reuses their existing building blocks (train_epoch,
test_epoch, prepare_trte_data, gen_trte_adj_mat, init_model_dict, init_optim,
save_model_dict) and adds a new function, train_test_with_history(), that
mirrors the structure of train_test() but records the loss_dict returned by
train_epoch() every epoch instead of discarding it.

Usage:
    from mogonet_training_logger import run_reps_with_history, plot_phase_histories

    pretrain_histories, finetune_histories = run_reps_with_history(
        data_folder="ROSMAP",
        view_list=[1, 2, 3],
        num_class=2,
        lr_e_pretrain=1e-3,
        lr_e=5e-4,
        lr_c=1e-3,
        num_epoch_pretrain=1500,
        num_epoch=500,
        num_reps=5,
        model_folder="ROSMAP/models",  # set to None to skip saving
    )

    plot_phase_histories(pretrain_histories, phase_name="Pretrain", title="ROSMAP pretrain (5 reps)")
    plot_phase_histories(finetune_histories, phase_name="Fine-tune", title="ROSMAP fine-tune (5 reps)")
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, save_model_dict
from train_test import prepare_trte_data, gen_trte_adj_mat, train_epoch, test_epoch

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

cuda = True if torch.cuda.is_available() else False


def train_test_with_history(data_folder, view_list, num_class,
                             lr_e_pretrain, lr_e, lr_c,
                             num_epoch_pretrain, num_epoch,
                             test_interval=50, model_folder=None):
    """
    Same training procedure as train_test(), but returns per-epoch loss
    history for both the pretrain and fine-tune phases instead of only
    printing progress.

    Returns:
        pretrain_history: dict {"epoch": [...], "C1": [...], "C2": [...], ...}
        finetune_history: dict {"epoch": [...], "C1": [...], ..., "C": [...],
                                 "test_epoch": [...], "test_acc": [...],
                                 "test_f1": [...], "test_auc": [...] (binary only)}
    """
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400, 400, 200]

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    # ---- Pretrain phase (VCDN off) ----
    pretrain_history = {"epoch": []}
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        loss_dict = train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                                 onehot_labels_tr_tensor, sample_weight_tr,
                                 model_dict, optim_dict, train_VCDN=False)
        pretrain_history["epoch"].append(epoch)
        for k, v in loss_dict.items():
            pretrain_history.setdefault(k, []).append(v)

    # ---- Fine-tune phase (VCDN on) ----
    finetune_history = {"epoch": []}
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch + 1):
        loss_dict = train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                                 onehot_labels_tr_tensor, sample_weight_tr,
                                 model_dict, optim_dict)
        finetune_history["epoch"].append(epoch)
        for k, v in loss_dict.items():
            finetune_history.setdefault(k, []).append(v)

        if epoch % test_interval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            finetune_history.setdefault("test_epoch", []).append(epoch)
            finetune_history.setdefault("test_acc", []).append(
                accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)))
            if num_class == 2:
                finetune_history.setdefault("test_f1", []).append(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)))
                finetune_history.setdefault("test_auc", []).append(
                    roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1]))
            else:
                finetune_history.setdefault("test_f1_weighted", []).append(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted'))
                finetune_history.setdefault("test_f1_macro", []).append(
                    f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro'))
            print("Test: Epoch {:d}  ACC {:.3f}".format(epoch, finetune_history["test_acc"][-1]))

    if model_folder is not None:
        os.makedirs(model_folder, exist_ok=True)
        save_model_dict(model_folder, model_dict)

    return pretrain_history, finetune_history


def run_reps_with_history(data_folder, view_list, num_class,
                           lr_e_pretrain, lr_e, lr_c,
                           num_epoch_pretrain, num_epoch,
                           num_reps=5, model_folder=None):
    """
    Runs train_test_with_history() num_reps times. If model_folder is given,
    each rep is saved to model_folder/<rep+1> (same convention as
    main_biomarker.py), so the saved models can be reused directly by
    cal_feat_imp() afterward.

    Returns:
        pretrain_histories: list of pretrain_history dicts, one per rep
        finetune_histories: list of finetune_history dicts, one per rep
    """
    pretrain_histories = []
    finetune_histories = []
    for rep in range(num_reps):
        print(f"\n=== Rep {rep + 1}/{num_reps} ===")
        rep_folder = os.path.join(model_folder, str(rep + 1)) if model_folder else None
        pretrain_hist, finetune_hist = train_test_with_history(
            data_folder, view_list, num_class,
            lr_e_pretrain, lr_e, lr_c,
            num_epoch_pretrain, num_epoch,
            model_folder=rep_folder,
        )
        pretrain_histories.append(pretrain_hist)
        finetune_histories.append(finetune_hist)
    return pretrain_histories, finetune_histories


def plot_phase_histories(histories, phase_name="", title=None, loss_keys=None):
    """
    Overlays loss curves from multiple reps on one figure: each rep plotted
    as a faint line, plus a bold mean curve per loss key.

    histories: list of history dicts (one per rep), as returned by
               train_test_with_history()
    loss_keys: which keys to plot (defaults to all keys except epoch/test_*)
    """
    if loss_keys is None:
        skip = {"epoch", "test_epoch", "test_acc", "test_f1", "test_auc",
                "test_f1_weighted", "test_f1_macro"}
        loss_keys = [k for k in histories[0].keys() if k not in skip]

    epochs = histories[0]["epoch"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_keys)))

    plt.figure(figsize=(8, 5))
    for key, color in zip(loss_keys, colors):
        # individual reps, faint
        for hist in histories:
            plt.plot(hist["epoch"], hist[key], color=color, alpha=0.25, linewidth=1)
        # mean across reps, bold
        stacked = np.array([hist[key] for hist in histories])
        plt.plot(epochs, stacked.mean(axis=0), color=color, linewidth=2.5, label=f"{key} (mean)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title or f"{phase_name} loss history ({len(histories)} reps)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example: ROSMAP, 5 reps
    pretrain_histories, finetune_histories = run_reps_with_history(
        data_folder="BRCA",
        view_list=[1, 2, 3],
        num_class=5,
        lr_e_pretrain=1e-3,
        lr_e=5e-4,
        lr_c=1e-3,
        num_epoch_pretrain=1500,
        num_epoch=500,
        num_reps=5,
        model_folder="BRCA/models",
    )

    plot_phase_histories(pretrain_histories, phase_name="Pretrain",
                          title="BRCA pretrain phase (5 reps)")
    plot_phase_histories(finetune_histories, phase_name="Fine-tune",
                          title="BRCA fine-tune phase (5 reps)")
