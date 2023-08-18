import numpy as np
import random
import warnings
from selectinf.Utils.discrete_family import discrete_family
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from .nbd_helpers import *
from .Utils.barrier_affine import solve_barrier_GGM


def _approx_log_reference(query_spec, grid, j0, k0, S_copy, n, p):
    """
    Approximate the log of the reference density on a grid.
    """

    """
    nonzero: is a pxp-dimensional vector, with the diagonal entries being 0
    """

    QS = query_spec
    nonzero = QS.nonzero
    active = add_diag(QS.active,val=0).astype(bool) # p x p with zero diagonal

    ref_hat = []
    solver = solve_barrier_GGM

    for k in range(grid.shape[0]):
        S_copy[j0, k0] = grid[k]
        S_copy[k0, j0] = grid[k]

        log_laplace_val = 0
        # TODO: Accomodate the ridge term in log_det
        sum_log_det = 0
        sum_laplace = 0

        for i in range(p):
            prec_rand_i = QS.prec_rands[i]
            nonzero_i = nonzero[i, :]
            active_i = active[i, :]

            non_void_selection = (active_i.sum() > 0)

            # Determine whether the Laplace approximation needs to be applied
            laplace_flag_i = False
            if i == j0 or i == k0:
                laplace_flag_i = True
            elif active_i[j0] or active_i[k0]:
                laplace_flag_i = True

            # Apply the Laplace approximation if the integral is not constant in s
            if laplace_flag_i and non_void_selection:
                #print("i:", i)
                #print("-i:", list(j for j in range(S_copy.shape[0]) if j != i))
                #print("E_i:", active_i)
                sum_laplace = sum_laplace + 1
                ## omega = a + B beta + c
                a_i = - S_copy[list(j for j in range(S_copy.shape[0]) if j != i), i] * n
                # TODO: Accomodate the ridge term in B_i
                B_i = ((S_copy[list(j for j in range(S_copy.shape[0]) if j != i)])[:,active_i] * n +
                       QS.ridge_terms[i] * np.eye(p-1)[:,QS.active[i,:]])
                c_i = QS.observed_subgrad[i,:]
                const_term = (a_i + c_i).T.dot(prec_rand_i).dot(a_i + c_i) / 2

                # Make sure selection information was saved correctly
                assert QS.linear_parts[i] is not None
                assert QS.offsets[i] is not None

                #print("i:", i)
                #print("linear:", QS.linear_parts[i])
                #print("offset:", QS.offsets[i])

                val, _, _ = solver(A=B_i, precision=prec_rand_i, c=a_i + c_i,
                                   feasible_point=QS.observed_soln[i,np.delete(active_i,i)],
                                   con_linear=QS.linear_parts[i],
                                   con_offset=QS.offsets[i])

                if np.isnan(val):
                    print("(",j0,",",k0,"), grid no.:", k, ", problem: ", i)

                log_laplace_val = log_laplace_val + (-val - const_term)


            # Determine whether the product of determinant needs to be calculated
            det_flag_i = False
            if i == j0 or i == k0:
                det_flag_i = False
            elif active_i[j0] and active_i[k0]: # or nonzero_i?
                det_flag_i = True

            if det_flag_i and non_void_selection:
                # TODO: Accomodate the ridge term in log_det
                # or nonzero_i?
                # print("S_copy[active_i, active_i].shape:", S_copy[active_i][:,active_i].shape)
                # sum_log_det = sum_log_det + np.log(np.abs(np.linalg.det(S_copy[active_i][:,active_i])))
                sum_log_det = sum_log_det + np.log(np.abs(np.linalg.det(S_copy[active_i][:, active_i] +
                                                                 np.eye(active_i.sum()) * QS.ridge_terms[i])))
                if np.isnan(sum_log_det):
                    print("detS+I:", np.linalg.det(S_copy[active_i][:, active_i] +
                                               np.eye(active_i.sum()) * QS.ridge_terms[i]))

        assert (log_laplace_val + sum_log_det).shape == ()
        ref_hat.append(log_laplace_val + sum_log_det)

        #print("#Laplace approx:", sum_laplace)

    return ref_hat

def approx_inference(j0k0, query_spec, X_n, n, p, ngrid=10000, ncoarse=None, level=0.9):
    j0 = j0k0[0]
    k0 = j0k0[1]
    # X_n: X / sqrt(n)
    S = X_n.T @ X_n

    inner_prod = S[j0,k0] # S = X.T X / n

    S_copy = np.copy(S)
    stat_grid = np.linspace(-3, 3, num=ngrid)

    if ncoarse is not None:
        coarse_grid = np.linspace(-3, 3, ncoarse)
        eval_grid = coarse_grid
    else:
        eval_grid = stat_grid
    ref_hat = _approx_log_reference(query_spec, eval_grid, j0, k0, S_copy, n, p)
    #print("ref_hat shape:", ref_hat.shape)

    def log_det_S_j_k(s_val):
        S_j_k = S_copy
        S_j_k[j0,k0] = s_val
        S_j_k[k0,j0] = s_val
        if np.linalg.det(S_j_k) < 0:
            #print("negative det", np.linalg.det(S_j_k),
            #      "grid", s_val)
            return -np.inf
        return np.log((np.linalg.det(S_j_k))) * (n-p-1)/2

    if ncoarse is None:
        logWeights = np.zeros((ngrid,))
        for g in range(ngrid):
            #print(logWeights[g])
            #print(log_det_S_j_k(eval_grid[g]))
            #print(ref_hat[g])
            logWeights[g] = log_det_S_j_k(eval_grid[g]) + ref_hat[g]

        # plt.plot(eval_grid, logWeights)

        # normalize logWeights
        logWeights = logWeights - np.max(logWeights)
        # Set extremely small values (< e^-500) to e^-500 for numerical stability
        # logWeights_zero = (logWeights < -500)
        # logWeights[logWeights_zero] = -500
        condlWishart = discrete_family(eval_grid,
                                       np.exp(logWeights),
                                       logweights=logWeights)
    else:
        # print("Coarse grid")
        approx_fn = interp1d(eval_grid,
                             ref_hat,
                             kind='quadratic',
                             bounds_error=False,
                             fill_value='extrapolate')
        grid = np.linspace(-3, 3, num=ngrid)
        logWeights = np.zeros((ngrid,))
        for g in range(ngrid):
            #print(log_det_S_j_k(grid[g]))
            #print(approx_fn(grid[g]))
            logWeights[g] = log_det_S_j_k(grid[g]) + approx_fn(grid[g])

        # plt.plot(grid, logWeights)

        # normalize logWeights
        logWeights = logWeights - np.max(logWeights)
        # Set extremely small values (< e^-500) to e^-500 for numerical stability
        # logWeights_zero = (logWeights < -500)
        # logWeights[logWeights_zero] = -500
        condlWishart = discrete_family(grid, np.exp(logWeights),
                                       logweights=logWeights)

    if np.isnan(logWeights).sum() != 0:
        print("logWeights contains nan")
    elif (logWeights == np.inf).sum() != 0:
        print("logWeights contains inf")
    elif (np.asarray(ref_hat) == np.inf).sum() != 0:
        print("ref_hat contains inf")
    elif (np.asarray(ref_hat) == -np.inf).sum() != 0:
        print("ref_hat contains -inf")
    elif np.isnan(np.asarray(ref_hat)).sum() != 0:
        print("ref_hat contains nan")

    neg_interval = condlWishart.equal_tailed_interval(observed=inner_prod,
                                                      alpha=1-level)
    if np.isnan(neg_interval[0]) or np.isnan(neg_interval[1]):
        print("Failed to construct intervals: nan")

    interval = invert_interval(neg_interval)

    pivot = condlWishart.ccdf(theta=0)

    return pivot, interval[0], interval[1]#neg_interval, condlWishart

