from __future__ import print_function

import sys
# For greatlakes simulations
sys.path.append('/home/yilingh/SI-Graphs')
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

def approx_inference_sim(X, prec, weights_const=1., ridge_const=0., randomizer_scale=1.,
                         parallel=False, ncores=4, logic = 'OR', ncoarse=200,
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
        intervals = nbd_instance.inference(parallel=parallel, ncoarse=ncoarse, ncores=ncores)
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


def nbd_simulations(m=3, proportion=0.5, logic_tf=0,
                    range_=range(0, 100), ncores=4, fix_np=True):
    # Encoding binary logic into str
    if logic_tf == 0:
        logic = 'OR'
    elif logic_tf == 1:
        logic = 'AND'

    # Operating characteristics
    oper_char = {}
    oper_char["n,p"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    oper_char["F1 score (post inf)"] = []
    oper_char["E size"] = []
    # oper_char["Selection power"] = []
    oper_char["Cond. power"] = []
    # oper_char["Power post inf"] = []
    oper_char["FDP"] = []

    if fix_np:
        np_list = [(400, 20)]
    else:
        np_list = [(200, 10), (400, 20), (1000, 50)]

    for np_pair in np_list:
        n = np_pair[0]
        p = np_pair[1]
        ## print(n, p)
        weights_const = 0.2
        ridge_const = 1.
        randomizer_scale = 1.
        ncoarse = 200

        for i in range(range_.start, range_.stop):
            n_instance = 0
            print(i)

            while True:  # run until we get some selection
                n_instance = n_instance + 1
                prec,cov,X = GGM_instance(n=n, p=p, max_edges=m, signal=0.7)
                n, p = X.shape
                # print((np.abs(prec) > 1e-5))
                noselection = False  # flag for a certain method having an empty selected set

                nonzero_n = naive_inference(X, prec, weights_const=weights_const,
                                            true_nonzero=None, logic=logic,
                                            solve_only=True, continued=False, nonzero_cont=None)

                nonzero_ds, subset_select = data_splitting(X, prec, weights_const=weights_const, proportion=proportion,
                                                           logic=logic, solve_only=True, continued=False,
                                                           nonzero_cont=None, subset_cont=None)

                noselection = np.min((nonzero_n.sum(), nonzero_ds.sum())) == 0
                if not noselection:
                    nonzero_approx, instance_approx = approx_inference_sim(X, prec, weights_const=weights_const,
                                                                           ridge_const=ridge_const,
                                                                           randomizer_scale=randomizer_scale,
                                                                           parallel=False, logic=logic,
                                                                           solve_only=True, continued=False,
                                                                           nbd_instance_cont=None)

                    noselection = np.min((nonzero_n.sum(), nonzero_ds.sum(), nonzero_approx.sum())) == 0

                # Continue with simultaneously nonzero instance
                if not noselection:
                    # Naive inference
                    true_non0 = (prec != 0)
                    for j in range(prec.shape[0]):
                        true_non0[j, j] = False
                    print("Naive")
                    nonzero_n, intervals_n, cov_rate_n, avg_len_n = naive_inference(X, prec,
                                                                                    weights_const=weights_const,
                                                                                    true_nonzero=None, logic=logic,
                                                                                    solve_only=False, continued=True,
                                                                                    nonzero_cont=nonzero_n)

                    # Data splitting
                    nonzero_ds, intervals_ds, cov_rate_ds, avg_len_ds = data_splitting(X, prec,
                                                                                       weights_const=weights_const,
                                                                                       proportion=proportion,
                                                                                       logic=logic, solve_only=False,
                                                                                       continued=True,
                                                                                       nonzero_cont=nonzero_ds,
                                                                                       subset_cont=subset_select)

                    print("|E|:", nonzero_approx.sum())
                    # Approximate inference
                    nonzero_approx, intervals_approx, cov_rate_approx, avg_len_approx \
                        = approx_inference_sim(X, prec, weights_const=weights_const,
                                               ridge_const=ridge_const, randomizer_scale=randomizer_scale,
                                               parallel=True, ncores=ncores,
                                               logic=logic, solve_only=False, continued=True,
                                               nbd_instance_cont=instance_approx, ncoarse=ncoarse)

                if not noselection:
                    # F1 scores
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


                    # Conditional Power post inference
                    cond_power_n = calculate_cond_power_graph(prec, selection=nonzero_n,
                                                             selection_CI=nonzero_n_int)
                    cond_power_ds = calculate_cond_power_graph(prec, selection=nonzero_ds,
                                                             selection_CI=nonzero_ds_int)
                    cond_power_approx = calculate_cond_power_graph(prec, selection=nonzero_approx,
                                                             selection_CI=nonzero_approx_int)

                    # FDP post inference
                    FDP_n = calculate_FDP_graph(beta_true=prec, selection=nonzero_n_int)
                    FDP_ds = calculate_FDP_graph(beta_true=prec, selection=nonzero_ds_int)
                    FDP_approx = calculate_FDP_graph(beta_true=prec, selection=nonzero_approx_int)


                    # Data splitting coverage
                    oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_ds.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_ds))
                    oper_char["avg length"].append(np.mean(avg_len_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["F1 score (post inf)"].append(F1_pi_ds)
                    oper_char["method"].append('Data Splitting')
                    oper_char["Cond. power"].append(cond_power_n)
                    oper_char["FDP"].append(FDP_n)

                    # Naive coverage
                    oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_n.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_n))
                    oper_char["avg length"].append(np.mean(avg_len_n))
                    oper_char["F1 score"].append(F1_n)
                    oper_char["F1 score (post inf)"].append(F1_pi_n)
                    oper_char["method"].append('Naive')
                    oper_char["Cond. power"].append(cond_power_ds)
                    oper_char["FDP"].append(FDP_ds)

                    # Approximate Inference coverage
                    oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_approx.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_approx))
                    oper_char["avg length"].append(np.mean(avg_len_approx))
                    oper_char["F1 score"].append(F1_approx)
                    oper_char["F1 score (post inf)"].append(F1_pi_approx)
                    oper_char["method"].append('Approx')
                    oper_char["Cond. power"].append(cond_power_approx)
                    oper_char["FDP"].append(FDP_approx)

                    print("# Instances needed for a non-null selection:", n_instance)

                    oper_char_df = pd.DataFrame.from_dict(oper_char)
                    if fix_np:
                        oper_char_df.to_csv('GGM_naive_ds_approx_logic' + str(logic_tf) + '_n400_p20_'
                                            + str(range_.start) + '_' + str(range_.stop) + '.csv', index=False)
                    else:
                        oper_char_df.to_csv('GGM_naive_ds_approx_logic' + str(logic_tf) + '_'
                                            + str(range_.start) + '_' + str(range_.stop) + '.csv', index=False)

                    break  # Go to next iteration if we have some selection

def nbd_simulations_vary_weights(m=2, proportion=0.5, logic_tf=0,
                                 range_=range(0, 100), ncores=4):
    # Encoding binary logic into str
    if logic_tf == 0:
        logic = 'OR'
    elif logic_tf == 1:
        logic = 'AND'

    # Operating characteristics
    oper_char = {}
    oper_char["weight const"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    oper_char["F1 score (post inf)"] = []
    oper_char["E size"] = []
    # oper_char["Selection power"] = []
    oper_char["Cond. power"] = []
    # oper_char["Power post inf"] = []
    oper_char["FDP"] = []

    weights = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for weights_const in weights:
        n = 200#np_pair[0]
        p = 10#np_pair[1]
        ## print(n, p)
        #weights_const = 0.2
        ridge_const = 1.
        randomizer_scale = 1.
        ncoarse = 200

        for i in range(range_.start, range_.stop):
            n_instance = 0
            print(i)

            while True:  # run until we get some selection
                n_instance = n_instance + 1
                prec,cov,X = GGM_instance(n=n, p=p, max_edges=m, signal=0.4)
                n, p = X.shape
                # print((np.abs(prec) > 1e-5))
                noselection = False  # flag for a certain method having an empty selected set

                nonzero_n = naive_inference(X, prec, weights_const=weights_const,
                                            true_nonzero=None, logic=logic,
                                            solve_only=True, continued=False, nonzero_cont=None)

                nonzero_ds, subset_select = data_splitting(X, prec, weights_const=weights_const, proportion=proportion,
                                                           logic=logic, solve_only=True, continued=False,
                                                           nonzero_cont=None, subset_cont=None)

                noselection = np.min((nonzero_n.sum(), nonzero_ds.sum())) == 0
                if not noselection:
                    nonzero_approx, instance_approx = approx_inference_sim(X, prec, weights_const=weights_const,
                                                                           ridge_const=ridge_const,
                                                                           randomizer_scale=randomizer_scale,
                                                                           parallel=False, logic=logic,
                                                                           solve_only=True, continued=False,
                                                                           nbd_instance_cont=None)

                    noselection = np.min((nonzero_n.sum(), nonzero_ds.sum(), nonzero_approx.sum())) == 0

                # Continue with simultaneously nonzero instance
                if not noselection:
                    print("|E^*|:", (prec != 0).sum())
                    # Naive inference
                    nonzero_n, intervals_n, cov_rate_n, avg_len_n = naive_inference(X, prec,
                                                                                    weights_const=weights_const,
                                                                                    true_nonzero=None, logic=logic,
                                                                                    solve_only=False, continued=True,
                                                                                    nonzero_cont=nonzero_n)
                    print("Naive |E|:", nonzero_n.sum())

                    # Data splitting
                    nonzero_ds, intervals_ds, cov_rate_ds, avg_len_ds = data_splitting(X, prec,
                                                                                       weights_const=weights_const,
                                                                                       proportion=proportion,
                                                                                       logic=logic, solve_only=False,
                                                                                       continued=True,
                                                                                       nonzero_cont=nonzero_ds,
                                                                                       subset_cont=subset_select)
                    print("Data Splitting |E|:", nonzero_ds.sum())

                    # Approximate inference
                    nonzero_approx, intervals_approx, cov_rate_approx, avg_len_approx \
                        = approx_inference_sim(X, prec, weights_const=weights_const,
                                               ridge_const=ridge_const, randomizer_scale=randomizer_scale,
                                               parallel=True, ncores=ncores,
                                               logic=logic, solve_only=False, continued=True,
                                               nbd_instance_cont=instance_approx, ncoarse=ncoarse)
                    print("Approx |E|:", nonzero_approx.sum())

                if not noselection:
                    # F1 scores
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


                    # Conditional Power post inference
                    cond_power_n = calculate_cond_power_graph(prec, selection=nonzero_n,
                                                             selection_CI=nonzero_n_int)
                    cond_power_ds = calculate_cond_power_graph(prec, selection=nonzero_ds,
                                                             selection_CI=nonzero_ds_int)
                    cond_power_approx = calculate_cond_power_graph(prec, selection=nonzero_approx,
                                                             selection_CI=nonzero_approx_int)

                    # FDP post inference
                    FDP_n = calculate_FDP_graph(beta_true=prec, selection=nonzero_n_int)
                    FDP_ds = calculate_FDP_graph(beta_true=prec, selection=nonzero_ds_int)
                    FDP_approx = calculate_FDP_graph(beta_true=prec, selection=nonzero_approx_int)


                    # Data splitting coverage
                    #oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_ds.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_ds))
                    oper_char["avg length"].append(np.mean(avg_len_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["F1 score (post inf)"].append(F1_pi_ds)
                    oper_char["method"].append('Data Splitting')
                    oper_char["Cond. power"].append(cond_power_n)
                    oper_char["FDP"].append(FDP_n)
                    oper_char["weight const"].append(weights_const)

                    # Naive coverage
                    #oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_n.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_n))
                    oper_char["avg length"].append(np.mean(avg_len_n))
                    oper_char["F1 score"].append(F1_n)
                    oper_char["F1 score (post inf)"].append(F1_pi_n)
                    oper_char["method"].append('Naive')
                    oper_char["Cond. power"].append(cond_power_ds)
                    oper_char["FDP"].append(FDP_ds)
                    oper_char["weight const"].append(weights_const)

                    # Approximate Inference coverage
                    #oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_approx.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_approx))
                    oper_char["avg length"].append(np.mean(avg_len_approx))
                    oper_char["F1 score"].append(F1_approx)
                    oper_char["F1 score (post inf)"].append(F1_pi_approx)
                    oper_char["method"].append('Approx')
                    oper_char["Cond. power"].append(cond_power_approx)
                    oper_char["FDP"].append(FDP_approx)
                    oper_char["weight const"].append(weights_const)

                    print("# Instances needed for a non-null selection:", n_instance)

                    oper_char_df = pd.DataFrame.from_dict(oper_char)
                    oper_char_df.to_csv('GGM_comp_logic1_' + str(logic_tf) + '_'
                                        + str(range_.start) + '_' + str(range_.stop) + '.csv', index=False)

                    break  # Go to next iteration if we have some selection

if __name__ == '__main__':
    argv = sys.argv
    # argv = [..., start, end, logic_tf, ncores]
    start, end = int(argv[1]), int(argv[2])
    logic_tf = int(argv[3])
    ncores = int(argv[4])
    nbd_simulations_vary_weights(range_=range(start, end), logic_tf=logic_tf,
                                 ncores=ncores)  # , logic_tf=logic_tf, s=s)