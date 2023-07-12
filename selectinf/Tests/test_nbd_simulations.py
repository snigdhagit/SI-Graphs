from __future__ import print_function

import numpy as np
import pandas as pd
import random
import nose.tools as nt
import collections
collections.Callable = collections.abc.Callable

from matplotlib import pyplot as plt
import seaborn as sns

from selectinf.nbd_lasso import nbd_lasso
from selectinf.Utils.discrete_family import discrete_family
from selectinf.Tests.instance import GGM_instance

from selectinf.Tests.nbd_naive_and_ds import *

def test_nbd_simulations(n=1000, p=30, max_edges=3, proportion=0.67,
                         n_iter=30):
    # Operating characteristics
    oper_char = {}
    oper_char["p"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    oper_char["E size"] = []

    for p in [10]:#, 20, 30]:
        for i in range(n_iter):
            n_instance = 0
            print(i)
            # np.random.seed(i)

            while True:  # run until we get some selection
                n_instance = n_instance + 1
                prec,cov,X = GGM_instance(n=200, p=p, max_edges=3)
                n, p = X.shape
                # print((np.abs(prec) > 1e-5))

                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    true_non0 = (prec!=0)
                    for j in range(prec.shape[0]):
                        true_non0[j,j] = False
                    nonzero_n, intervals_n, cov_rate_n, avg_len_n = naive_inference(X, prec,
                                                                                    weights=0.15, true_nonzero = true_non0)
                    noselection = (nonzero_n is None)
                    # print(nonzero_n)
                    # print(nonzero_n.shape)

                if not noselection:
                    nonzero_ds, intervals_ds, cov_rate_ds, avg_len_ds = data_splitting(X, prec, weights=0.15,
                                                                                       proportion=proportion)
                    noselection = (nonzero_ds is None)
                    # print(nonzero_ds.shape)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    print("symmetric nonzero:", is_sym(nonzero_n))
                    F1_n = calculate_F1_score_graph(prec, selection=nonzero_n)
                    F1_ds = calculate_F1_score_graph(prec, selection=nonzero_ds)

                    # Data splitting coverage
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_ds.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_ds))
                    oper_char["avg length"].append(np.mean(avg_len_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data Splitting')

                    # Naive coverage
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_n.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_n))
                    oper_char["avg length"].append(np.mean(avg_len_n))
                    oper_char["F1 score"].append(F1_n)
                    oper_char["method"].append('Naive')

                    print("# Instances needed for a non-null selection:", n_instance)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('GGM_naive_ds.csv', index=False)

    print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['p', 'method']).mean())

    sns.boxplot(y=oper_char_df["coverage rate"],
                x=oper_char_df["p"],
                hue=oper_char_df["method"],
                showmeans=True,
                orient="v")
    plt.show()

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["p"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    # len_plot.set_ylim(5, 17)
    plt.show()

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                          x=oper_char_df["p"],
                          hue=oper_char_df["method"],
                          showmeans=True,
                          orient="v")
    F1_plot.set_ylim(0, 1)
    plt.show()

    size_plot = sns.boxplot(y=oper_char_df["E size"],
                          x=oper_char_df["p"],
                          hue=oper_char_df["method"],
                          showmeans=True,
                          orient="v")
    plt.show()

def test_plotting(path='GGM_naive_ds.csv'):
    oper_char_df = pd.read_csv(path)
    #sns.histplot(oper_char_df["sparsity size"])
    #plt.show()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(12,6))

    print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['p', 'method']).mean())

    cov_plot = sns.boxplot(y=oper_char_df["coverage rate"],
                           x=oper_char_df["p"],
                           hue=oper_char_df["method"],
                           palette="pastel",
                           orient="v", ax=ax1,
                           showmeans=True,
                           linewidth=1)
    cov_plot.set(title='Coverage')
    cov_plot.set_ylim(0., 1.05)
    #plt.tight_layout()
    cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
    #ax1.set_ylabel("")  # remove y label, but keep ticks

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["p"],
                           hue=oper_char_df["method"],
                           palette="pastel",
                           orient="v", ax=ax2,
                           linewidth=1)
    len_plot.set(title='Length')
    # len_plot.set_ylim(0, 100)
    # len_plot.set_ylim(3.5, 7.8)
    # plt.tight_layout()
    # ax2.set_ylabel("")  # remove y label, but keep ticks

    handles, labels = ax2.get_legend_handles_labels()
    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
    fig.subplots_adjust(bottom=0.2)
    fig.legend(handles, labels, loc='lower center', ncol=4)

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                          x=oper_char_df["p"],
                          hue=oper_char_df["method"],
                          palette="pastel",
                          orient="v", ax=ax3,
                          linewidth=1)
    F1_plot.set(title='F1 score')

    size_plot = sns.boxplot(y=oper_char_df["E size"],
                          x=oper_char_df["p"],
                          hue=oper_char_df["method"],
                          palette="pastel",
                          orient="v", ax=ax4,
                          linewidth=1)
    size_plot.set(title='|E|')

    cov_plot.legend_.remove()
    len_plot.legend_.remove()
    F1_plot.legend_.remove()
    size_plot.legend_.remove()

    plt.suptitle("Naive and Data Splitting (0.67)")
    plt.show()