import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from train_test import prepare_trte_data
from train_test import gen_trte_adj_mat


if __name__ == "__main__":
    adj_parameter = 2
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data("ROSMAP", [1, 2, 3])
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    # print(data_tr_list[0].cpu().numpy())
    
    # df_data_tr_list = [mtx.cpu().numpy() for mtx in data_tr_list]
    # omics = ["mRNA", "methDNA", "miRNA"]
    # for df, omic in zip(df_data_tr_list, omics):
    #     np.savetxt(f"{omic}.csv", df, delimiter=",", fmt="%.2f")

    # print(len(data_trte_list[0][0]))
    # print(trte_idx["tr"])
    # print(labels_trte[trte_idx["tr"]])
    # for list in adj_tr_list:
    #     print(list)

    dense_adj_tr_list = [mtx.to_dense().cpu().numpy() for mtx in adj_tr_list]
    sns.heatmap(dense_adj_tr_list[0], cmap="Blues")
    plt.savefig("mrna_adj.png", dpi=220, bbox_inches="tight")
    plt.close()
    sns.heatmap(dense_adj_tr_list[1], cmap="Greens")
    plt.savefig("methdna_adj.png", dpi=220, bbox_inches="tight")
    plt.close()
    sns.heatmap(dense_adj_tr_list[2], cmap="Oranges")
    plt.savefig("mirna_adj.png", dpi=220, bbox_inches="tight")
    plt.close()
