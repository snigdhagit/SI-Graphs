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


def nbd_simulations_vary_omega(s=4, logic_tf=1, range_=range(0, 100)):
    # Encoding binary logic into str
    if logic_tf == 0:
        logic = 'OR'
    elif logic_tf == 1:
        logic = 'AND'

    # Operating characteristics
    oper_char = {}
    oper_char["n,p"] = []
    oper_char["randomizer scale"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    oper_char["F1 score (post inf)"] = []
    oper_char["E size"] = []


    for np_pair in [(200, 10), (400, 20), (1000, 50)]:  # , 20, 30]:
        n = np_pair[0]
        p = np_pair[1]
        print(n, p)
        for i in range(range_.start, range_.stop):
            n_instance = 0
            print(i)
            # np.random.seed(i)

            tau = [1, 2, 5]

            while True:  # run until we get some selection
                n_instance = n_instance + 1
                prec,cov,X = GGM_instance(n=n, p=p, max_edges=s)
                n, p = X.shape
                # print((np.abs(prec) > 1e-5))
                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    nonzero_1, intervals_1, cov_rate_1, avg_len_1 \
                        = approx_inference_sim(X, prec, weights_const=0.5,
                                               ridge_const=1., randomizer_scale=tau[0],
                                               parallel=False, logic=logic, ncoarse=200)
                    noselection = (nonzero_1 is None)

                if not noselection:
                    nonzero_2, intervals_2, cov_rate_2, avg_len_2 \
                        = approx_inference_sim(X, prec, weights_const=0.5,
                                               ridge_const=1., randomizer_scale=tau[1],
                                               parallel=False, logic=logic, ncoarse=200)
                    noselection = (nonzero_2 is None)

                if not noselection:
                    nonzero_3, intervals_3, cov_rate_3, avg_len_3 \
                        = approx_inference_sim(X, prec, weights_const=0.5,
                                               ridge_const=1., randomizer_scale=tau[2],
                                               parallel=False, logic=logic, ncoarse=200)
                    noselection = (nonzero_3 is None)
                    # print(nonzero_ds.shape)

                if not noselection:
                    # F1 scores
                    # Post-inference selection
                    nonzero_1_int = interval_selection(intervals_1, nonzero_1)
                    nonzero_2_int = interval_selection(intervals_2, nonzero_2)
                    nonzero_3_int = interval_selection(intervals_3, nonzero_3)

                    # Selection F1-score
                    F1_1 = calculate_F1_score_graph(prec, selection=nonzero_1)
                    F1_2 = calculate_F1_score_graph(prec, selection=nonzero_2)
                    F1_3 = calculate_F1_score_graph(prec, selection=nonzero_3)

                    # Post-inference F1-score
                    F1_pi_1 = calculate_F1_score_graph(prec, selection=nonzero_1_int)
                    F1_pi_2 = calculate_F1_score_graph(prec, selection=nonzero_2_int)
                    F1_pi_3 = calculate_F1_score_graph(prec, selection=nonzero_3_int)


                    # First scale coverage
                    oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_1.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_1))
                    oper_char["avg length"].append(np.mean(avg_len_1))
                    oper_char["F1 score"].append(F1_1)
                    oper_char["F1 score (post inf)"].append(F1_pi_1)
                    oper_char["randomizer scale"].append(tau[0])

                    # Second scale coverage
                    oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_2.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_2))
                    oper_char["avg length"].append(np.mean(avg_len_2))
                    oper_char["F1 score"].append(F1_2)
                    oper_char["F1 score (post inf)"].append(F1_pi_2)
                    oper_char["randomizer scale"].append(tau[1])

                    # Third Inference coverage
                    oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")
                    oper_char["E size"].append(nonzero_3.sum())
                    oper_char["coverage rate"].append(np.mean(cov_rate_3))
                    oper_char["avg length"].append(np.mean(avg_len_3))
                    oper_char["F1 score"].append(F1_3)
                    oper_char["F1 score (post inf)"].append(F1_pi_3)
                    oper_char["randomizer scale"].append(tau[2])

                    print("# Instances needed for a non-null selection:", n_instance)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('GGM_naive_ds_approx_vary_omega' + str(range_.start) + '_' + str(range_.stop) + '.csv', index=False)


if __name__ == '__main__':
    argv = sys.argv
    # argv = [..., start, end, logic_tf, s]
    start, end = 0,10#int(argv[1]), int(argv[2])
    # logic_tf = int(argv[3])
    #s = int(argv[4])
    # print("start:", start, ", end:", end)
    nbd_simulations_vary_omega(range_=range(start, end))#, logic_tf=logic_tf, s=s)