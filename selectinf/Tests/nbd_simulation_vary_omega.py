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


def nbd_simulations_vary_omega(m=3, logic_tf=1, range_=range(0, 100),
                               ncores=4, fix_np=True):
    # Encoding binary logic into str
    if logic_tf == 0:
        logic = 'OR'
    elif logic_tf == 1:
        logic = 'AND'

    # Operating characteristics
    oper_char = {}
    oper_char["n,p"] = []
    oper_char["randomizer variance"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["F1 score"] = []
    oper_char["F1 score (post inf)"] = []
    oper_char["E size"] = []
    oper_char["Selection power"] = []
    oper_char["Cond. power"] = []
    oper_char["Power post inf"] = []
    oper_char["FDP"] = []

    if fix_np:
        np_list = [(400, 20)]
    else:
        np_list = [(200, 10), (400, 20), (1000, 50)]

    for np_pair in np_list:
        n = np_pair[0]
        p = np_pair[1]
        print(n, p)
        for i in range(range_.start, range_.stop):
            n_instance = 0
            print(i)
            weights_const = 0.2
            ridge_const = 1
            ncoarse = 500

            # np.random.seed(i)

            # Vary randomizer scale from 0.5 to 5 on an equi-spaced grid
            tau_sq = np.array([0.5, 1, 2, 3, 4])

            while True:  # run until we get some selection
                n_instance = n_instance + 1
                prec,cov,X = GGM_instance(n=n, p=p, max_edges=m, signal=0.7)
                n, p = X.shape
                # print((np.abs(prec) > 1e-5))
                noselection = False  # flag for a certain method having an empty selected set

                nonzeros = []
                instances = []
                nonzero_sums = []

                for k in range(tau_sq.shape[0]):
                    nonzero_k, instance_k = approx_inference_sim(X, prec, weights_const=weights_const,
                                                                 ridge_const=ridge_const,
                                                                 randomizer_scale=np.sqrt(tau_sq[k]),
                                                                 parallel=False,
                                                                 logic=logic, solve_only=True, continued=False,
                                                                 nbd_instance_cont=None)
                    nonzeros.append(nonzero_k)
                    instances.append(instance_k)
                    nonzero_sums.append(nonzero_k.sum())
                    print("tau sq:", tau_sq[k], "|E|:", nonzero_k.sum(),
                          "|E^*|:", (prec !=0).sum())

                noselection = np.min(nonzero_sums) == 0

                if not noselection:
                    nonzeros = []
                    intervals = []
                    cov_rates = []
                    avg_lens = []
                    for k in range(tau_sq.shape[0]):
                        if not noselection:
                            nonzero_k, intervals_k, cov_rate_k, avg_len_k \
                                = approx_inference_sim(X, prec, weights_const=weights_const,
                                                       ridge_const=ridge_const, randomizer_scale=np.sqrt(tau_sq[k]),
                                                       parallel=True, ncores=ncores,
                                                       logic=logic, solve_only=False, continued=True,
                                                       nbd_instance_cont=instances[k], ncoarse=ncoarse)
                            print("tau sq:", tau_sq[k], "inference done")
                            nonzeros.append(nonzero_k)
                            intervals.append(intervals_k)
                            cov_rates.append(cov_rate_k)
                            avg_lens.append(avg_len_k)
                            noselection = (nonzero_k is None)

                if not noselection:
                    # Collect values shared across scales
                    # First scale coverage
                    for k in range(tau_sq.shape[0]):
                        oper_char["n,p"].append("(" + str(n) + "," + str(p) + ")")

                    # F1 scores
                    # Post-inference selection
                    nonzero_ints = []
                    for k in range(tau_sq.shape[0]):
                        nonzero_k_int = interval_selection(intervals[k], nonzeros[k])
                        nonzero_ints.append(nonzero_k_int)

                    # Selection F1-score
                    F1s = []
                    for k in range(tau_sq.shape[0]):
                        F1_k = calculate_F1_score_graph(prec, selection=nonzeros[k])
                        F1s.append(F1_k)

                    # Post-inference F1-score
                    F1_pis = []
                    for k in range(tau_sq.shape[0]):
                        F1_pi_k = calculate_F1_score_graph(prec, selection=nonzero_ints[k])
                        F1_pis.append(F1_pi_k)

                    # Conditional Power post inference
                    cond_powers = []
                    for k in range(tau_sq.shape[0]):
                        cond_power_k = calculate_cond_power_graph(prec, selection=nonzeros[k],
                                                                 selection_CI=nonzero_ints[k])
                        cond_powers.append(cond_power_k)

                    # FDP post inference
                    FDPs = []
                    for k in range(tau_sq.shape[0]):
                        FDPk = calculate_FDP_graph(beta_true=prec, selection=nonzero_ints[k])
                        FDPs.append(FDPk)

                    # Selection power
                    sel_power = []
                    for k in range(tau_sq.shape[0]):
                        sel_power_k = calculate_power_graph(beta_true=prec, selection=nonzeros[k])
                        sel_power.append(sel_power_k)

                    # Power post-inference
                    power_pis = []
                    for k in range(tau_sq.shape[0]):
                        power_pi_k = calculate_power_graph(beta_true=prec, selection=nonzero_ints[k])
                        power_pis.append(power_pi_k)

                    for k in range(tau_sq.shape[0]):
                        oper_char["randomizer variance"].append(tau_sq[k])
                        oper_char["E size"].append(nonzeros[k].sum())
                        oper_char["coverage rate"].append(np.mean(cov_rates[k]))
                        oper_char["avg length"].append(np.mean(avg_lens[k]))
                        oper_char["F1 score"].append(F1s[k])
                        oper_char["F1 score (post inf)"].append(F1_pis[k])
                        oper_char["Cond. power"].append(cond_powers[k])
                        oper_char["FDP"].append(FDPs[k])
                        oper_char["Selection power"].append(sel_power[k])
                        oper_char["Power post inf"].append(power_pis[k])

                    print("# Instances needed for a non-null selection:", n_instance)

                    # Save results to avoid losing info
                    oper_char_df = pd.DataFrame.from_dict(oper_char)
                    oper_char_df.to_csv('GGM_vary_omega_logic' + str(logic_tf) + '_'
                                        + str(range_.start) + '_' + str(range_.stop) + '.csv', index=False)
                    break  # Go to next iteration if we have some selection



if __name__ == '__main__':
    argv = sys.argv
    # argv = [..., start, end, logic_tf, s]
    start, end = int(argv[1]), int(argv[2])
    logic_tf = int(argv[3])
    ncores = int(argv[4])
    fixnp = int(argv[5])
    if fixnp == 1:
        fixnp = True
    else:
        fixnp = False
    #s = int(argv[4])
    # print("start:", start, ", end:", end)
    nbd_simulations_vary_omega(range_=range(start, end), logic_tf=logic_tf,
                               ncores=ncores, fix_np=fixnp)#argv[3])#, logic_tf=logic_tf, s=s)