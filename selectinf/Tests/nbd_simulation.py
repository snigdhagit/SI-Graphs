from __future__ import print_function

import numpy as np
import pandas as pd
import random
import nose.tools as nt
import collections
collections.Callable = collections.abc.Callable
import sys
# For greatlakes simulations
sys.path.append('/home/yilingh/SI-Graphs')

from matplotlib import pyplot as plt
import seaborn as sns

from selectinf.nbd_lasso import nbd_lasso
from selectinf.Utils.discrete_family import discrete_family
from selectinf.Tests.instance import GGM_instance

from selectinf.Tests.nbd_naive_and_ds import *

def approx_inference_sim(X, prec, weights_const=1., ridge_const=0., randomizer_scale=1.,
                         parallel=False, logic = 'OR', ncoarse=200,
                         solve_only=False, continued=False, nbd_instance_cont=None):
    # Precision matrix is in its original order, not scaled by root n
    # X is also in its original order
    n,p = X.shape

    if not continued:
        nbd_instance = nbd_lasso.gaussian(X, n_scaled=False, weights_const=weights_const,
                                          ridge_terms=ridge_const, randomizer_scale=randomizer_scale)
        active_signs_random = nbd_instance.fit(logic=logic)
        nonzero = nbd_instance.nonzero

    # If we only need to solve the Lasso
    if solve_only:
        return nonzero, nbd_instance

    # If we continue a previous run with a nontrivial selection
    if continued:
        nbd_instance = nbd_instance_cont
        nonzero = nbd_instance.nonzero
        assert nonzero.sum() > 0

    # Construct intervals
    if nonzero.sum() > 0:
        # Intervals returned is in its original (unscaled) order
        intervals = nbd_instance.inference(parallel=parallel, ncoarse=ncoarse)
        # coverage is upper-triangular
        coverage = get_coverage(nonzero, intervals, prec, n, p, scale=False)
        interval_len = 0
        nonzero_count = 0  # nonzero_count is essentially upper-triangular
        for i in range(p):
            for j in range(i+1,p):
                if nonzero[i,j]:
                    interval = intervals[i,j,:]
                    interval_len = interval_len + (interval[1] - interval[0])
                    nonzero_count = nonzero_count + 1
        avg_len = interval_len / nonzero_count
        cov_rate = coverage.sum() / nonzero_count
        return nonzero, intervals, cov_rate, avg_len
    return None, None, None, None


def nbd_simulations(n=1000, p=30, s=4, proportion=0.5, logic_tf=0,
                    range_=range(0, 100)):
    # Encoding binary logic into str
    if logic_tf == 0:
        logic = 'OR'
    elif logic_tf == 1:
        logic = 'AND'

    # Operating characteristics
    oper_char = {}
    oper_char["p"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    oper_char["F1 score (post inf)"] = []
    oper_char["E size"] = []

    for p in [10, 20, 50]:
        for i in range(range_.start, range_.stop):
            n_instance = 0
            # print(i)
            # np.random.seed(i)

            while True:  # run until we get some selection
                n_instance = n_instance + 1
                prec,cov,X = GGM_instance(n=200, p=p, max_edges=s)
                n, p = X.shape
                # print((np.abs(prec) > 1e-5))
                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    true_non0 = (prec!=0)
                    for j in range(prec.shape[0]):
                        true_non0[j,j] = False
                    print("Naive")
                    nonzero_n, intervals_n, cov_rate_n, avg_len_n\
                        = naive_inference(X, prec, weights_const=0.5,
                                          logic=logic)
                    noselection = (nonzero_n is None)

                if not noselection:
                    print("DS")
                    nonzero_ds, intervals_ds, cov_rate_ds, avg_len_ds\
                        = data_splitting(X, prec, weights_const=0.5,
                                         proportion=proportion, logic=logic)
                    noselection = (nonzero_ds is None)
                    if not noselection:
                        print("DS Length:", avg_len_ds)

                if not noselection:
                    print("Approx")
                    nonzero_approx, intervals_approx, cov_rate_approx, avg_len_approx \
                        = approx_inference_sim(X, prec, weights_const=0.5,
                                               ridge_const=1., randomizer_scale=1.,
                                               parallel=False, logic=logic)
                    noselection = (nonzero_approx is None)
                    # print(nonzero_ds.shape)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    print("symmetric nonzero:", is_sym(nonzero_n))
                    # Post-inference selection
                    nonzero_n_int = interval_selection(intervals_n, nonzero_n)
                    nonzero_ds_int = interval_selection(intervals_ds, nonzero_ds)
                    nonzero_approx_int = interval_selection(intervals_approx, nonzero_approx)

                    # Selection F1-score
                    F1_n = calculate_F1_score_graph(prec, selection=nonzero_n)
                    F1_ds = calculate_F1_score_graph(prec, selection=nonzero_ds)
                    F1_approx = calculate_F1_score_graph(prec, selection=nonzero_approx)

                    # Post-inference F1-score
                    F1_pi_n = calculate_F1_score_graph(prec, selection=nonzero_n_int)
                    F1_pi_ds = calculate_F1_score_graph(prec, selection=nonzero_ds_int)
                    F1_pi_approx = calculate_F1_score_graph(prec, selection=nonzero_approx_int)


                    # Data splitting coverage
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_ds.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_ds))
                    oper_char["avg length"].append(np.mean(avg_len_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["F1 score (post inf)"].append(F1_pi_ds)
                    oper_char["method"].append('Data Splitting')

                    # Naive coverage
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_n.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_n))
                    oper_char["avg length"].append(np.mean(avg_len_n))
                    oper_char["F1 score"].append(F1_n)
                    oper_char["F1 score (post inf)"].append(F1_pi_n)
                    oper_char["method"].append('Naive')

                    # Approximate Inference coverage
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_approx.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_approx))
                    oper_char["avg length"].append(np.mean(avg_len_approx))
                    oper_char["F1 score"].append(F1_approx)
                    oper_char["F1 score (post inf)"].append(F1_pi_approx)
                    oper_char["method"].append('Approx')

                    print("# Instances needed for a non-null selection:", n_instance)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('GGM_naive_ds_approx' + str(range_.start) + '_' + str(range_.stop) + '.csv', index=False)

def nbd_simulations_logic(n=1000, p=30, max_edges=3, proportion=0.5, range_=range(0, 100)):
    # Operating characteristics
    oper_char = {}
    oper_char["p"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    oper_char["F1 score (post inf)"] = []
    oper_char["E size"] = []

    for p in [10, 20, 50]:
        weights_const = 0.75
        ridge_const = 1.
        randomizer_scale = 1.
        ncoarse=int(200)

        for i in range_:
            n_instance = 0
            print(i)
            # np.random.seed(i)

            while True:  # run until we get some selection
                n_instance = n_instance + 1
                prec,cov,X = GGM_instance(n=1000, p=p, max_edges=4)
                n, p = X.shape
                # print((np.abs(prec) > 1e-5))
                noselection = False  # flag for a certain method having an empty selected set

                nonzero_n = naive_inference(X, prec, weights_const=weights_const,
                                            true_nonzero=None, logic='OR',
                                            solve_only=True, continued=False, nonzero_cont=None)

                nonzero_ds, subset_select = data_splitting(X, prec, weights_const=weights_const, proportion=proportion,
                                                           logic='OR', solve_only=True, continued=False,
                                                           nonzero_cont=None, subset_cont=None)

                noselection = np.min((nonzero_n.sum(), nonzero_ds.sum())) == 0

                if not noselection:
                    nonzero_AND, instance_AND = approx_inference_sim(X, prec, weights_const=weights_const,
                                                                           ridge_const=ridge_const,
                                                                           randomizer_scale=randomizer_scale,
                                                                           parallel=False, logic = 'AND',
                                                                           solve_only=True, continued=False,
                                                                           nbd_instance_cont=None)

                    noselection = np.min((nonzero_n.sum(), nonzero_ds.sum(), nonzero_AND.sum())) == 0

                if not noselection:
                    nonzero_OR, instance_OR = approx_inference_sim(X, prec, weights_const=weights_const,
                                                                           ridge_const=ridge_const,
                                                                           randomizer_scale=randomizer_scale,
                                                                           parallel=False, logic = 'OR',
                                                                           solve_only=True, continued=False,
                                                                           nbd_instance_cont=None)

                    noselection = (nonzero_OR.sum() == 0)

                # Continue with simultaneously nonzero instance
                if not noselection:
                    # Naive inference
                    true_non0 = (prec!=0)
                    for j in range(prec.shape[0]):
                        true_non0[j,j] = False
                    print("Naive")
                    nonzero_n, intervals_n, cov_rate_n, avg_len_n = naive_inference(X, prec,
                                                                                    weights_const=weights_const,
                                                                                    true_nonzero=None, logic='OR',
                                                                                    solve_only=False, continued=True,
                                                                                    nonzero_cont=nonzero_n)

                    # Data splitting
                    nonzero_ds, intervals_ds, cov_rate_ds, avg_len_ds = data_splitting(X, prec,
                                                                                       weights_const=weights_const,
                                                                                       proportion=proportion,
                                                                                       logic='OR', solve_only=False,
                                                                                       continued=True,
                                                                                       nonzero_cont=nonzero_ds,
                                                                                       subset_cont=subset_select)

                    # Approximate inference with AND
                    nonzero_AND, intervals_AND, cov_rate_AND, avg_len_AND \
                        = approx_inference_sim(X, prec, weights_const=weights_const,
                                               ridge_const=ridge_const, randomizer_scale=randomizer_scale,
                                               parallel=False, logic = 'AND', solve_only=False, continued=True,
                                               nbd_instance_cont=instance_AND, ncoarse=ncoarse)
                    # Approximate inference with OR
                    nonzero_OR, intervals_OR, cov_rate_OR, avg_len_OR \
                        = approx_inference_sim(X, prec, weights_const=weights_const,
                                               ridge_const=ridge_const, randomizer_scale=randomizer_scale,
                                               parallel=False, logic='OR', solve_only=False, continued=True,
                                               nbd_instance_cont=instance_OR, ncoarse=ncoarse)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    print("symmetric nonzero:", is_sym(nonzero_n))
                    # Post-inference selection
                    nonzero_n_int = interval_selection(intervals_n, nonzero_n)
                    nonzero_ds_int = interval_selection(intervals_ds, nonzero_ds)
                    nonzero_AND_int = interval_selection(intervals_AND, nonzero_AND)
                    nonzero_OR_int = interval_selection(intervals_OR, nonzero_OR)

                    # Selection F1-score
                    F1_n = calculate_F1_score_graph(prec, selection=nonzero_n)
                    F1_ds = calculate_F1_score_graph(prec, selection=nonzero_ds)
                    F1_AND = calculate_F1_score_graph(prec, selection=nonzero_AND)
                    F1_OR = calculate_F1_score_graph(prec, selection=nonzero_OR)

                    # Post-inference F1-score
                    F1_pi_n = calculate_F1_score_graph(prec, selection=nonzero_n_int)
                    F1_pi_ds = calculate_F1_score_graph(prec, selection=nonzero_ds_int)
                    F1_pi_AND = calculate_F1_score_graph(prec, selection=nonzero_AND_int)
                    F1_pi_OR = calculate_F1_score_graph(prec, selection=nonzero_OR_int)

                    # Data splitting coverage
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_ds.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_ds))
                    oper_char["avg length"].append(np.mean(avg_len_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["F1 score (post inf)"].append(F1_pi_ds)
                    oper_char["method"].append('Data Splitting')

                    # Naive coverage
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_n.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_n))
                    oper_char["avg length"].append(np.mean(avg_len_n))
                    oper_char["F1 score"].append(F1_n)
                    oper_char["F1 score (post inf)"].append(F1_pi_n)
                    oper_char["method"].append('Naive')

                    # Approximate Inference coverage with AND
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_AND.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_AND))
                    oper_char["avg length"].append(np.mean(avg_len_AND))
                    oper_char["F1 score"].append(F1_AND)
                    oper_char["F1 score (post inf)"].append(F1_pi_AND)
                    oper_char["method"].append('Approx (AND)')
                    # Approximate Inference coverage with OR
                    oper_char["p"].append(p)
                    oper_char["E size"].append(nonzero_AND.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_OR))
                    oper_char["avg length"].append(np.mean(avg_len_OR))
                    oper_char["F1 score"].append(F1_OR)
                    oper_char["F1 score (post inf)"].append(F1_pi_OR)
                    oper_char["method"].append('Approx (OR)')

                    print("# Instances needed for a non-null selection:", n_instance)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('GGM_naive_ds_approx_andor' + str(range_.start) + '_' + str(range_.stop) + '.csv', index=False)

if __name__ == '__main__':
    argv = sys.argv
    # argv = [..., start, end, logic_tf, s]
    start, end = int(argv[1]), int(argv[2])
    # logic_tf = int(argv[3])
    #s = int(argv[4])
    # print("start:", start, ", end:", end)
    nbd_simulations(range_=range(start, end))#, logic_tf=logic_tf, s=s)