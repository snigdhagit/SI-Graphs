import numpy as np
import random
import warnings
from selectinf.nbd_lasso import nbd_lasso
from selectinf.Utils.discrete_family import discrete_family
import matplotlib.pyplot as plt
from scipy.integrate import quad
from timebudget import timebudget
from scipy.optimize import root_scalar

## Helper functions that manages the graph
def remove_diag(A):
    p = A.shape[0]
    A_new = np.zeros((p,p-1))
    for i in range(p):
        A_new[i] = np.delete(A[i],i)
    return A_new

def add_diag(A,val):
    p = A.shape[0]
    A_new = np.zeros((p,p))
    for i in range(p):
        A_new[i,0:i] = A[i,0:i]
        A_new[i,i] = val
        A_new[i,i+1:p] = A[i,i:p-1]
    return A_new

def is_sym(A, tol = 1e-8):
    A = A.astype(float)
    return(np.max(np.abs(A-A.T)) < tol)

def invert_interval(interval, scalar=1.):
    interval_new = ((interval[1]*-1) / scalar, interval[0]*-1 / scalar)
    return interval_new

## Functions for numerical calculations
def find_root(f, y, lb, ub, tol=1e-6):
    """
    searches for solution to f(x) = y in (lb, ub), where
    f is a monotone decreasing function
    """

    # make sure solution is in range
    a, b = lb, ub
    fa, fb = f(a), f(b)

    # assume a < b
    if fa > y and fb > y:
        while fb > y:
            b, fb = b + (b - a), f(b + (b - a))
    elif fa < y and fb < y:
        while fa < y:
            a, fa = a - (b - a), f(a - (b - a))

    # determine the necessary number of iterations
    try:
        max_iter = int(np.ceil((np.log(tol) - np.log(b - a)) / np.log(0.5)))
    except OverflowError:
        warnings.warn('root finding failed, returning np.nan')
        return np.nan

    # bisect (slow but sure) until solution is obtained
    for _ in range(max_iter):
        try:
            c, fc = (a + b) / 2, f((a + b) / 2)
            if fc > y:
                a = c
            elif fc < y:
                b = c
        except OverflowError:
            warnings.warn('root finding failed, returning np.nan')
            return np.nan

    return c

## Functions used in inference
def bootstrap_variance(X, b_max=500):
    n,p = X.shape
    S_boot = np.zeros((b_max, p, p))
    for b in range(b_max):
        sample_idx = np.random.choice(range(n),replace=True,size=n)
        X_b = X[sample_idx]
        S_boot[b,:,:] = X_b.T @ X_b
    # Upper-triangular!
    inner_vars = np.zeros((p, p))
    for i in range(p):
        for j in range(i+1,p):
            S_ij_sample = S_boot[:,i,j]
            inner_vars[i,j] = np.var(S_ij_sample)
    return inner_vars

def edge_inference(j0k0, S, n, p, var=None,
                   ngrid=10000):
    j0 = j0k0[0]
    k0 = j0k0[1]
    # n_total: the total data points in data splitting
    #        : the raw dimension of X in naive
    inner_prod = S[j0,k0]
    # print("inner_prod", "(", j0, ",", k0, "):" , inner_prod)
    # print("var:", var)

    S_copy = np.copy(S)

    #stat_grid = np.zeros((ngrid,))
    #print("n=100 assumed")
    stat_grid = np.linspace(-10,10,num=ngrid)
    def log_det_S_j_k(s_val):
        S_j_k = S_copy
        S_j_k[j0,k0] = s_val
        S_j_k[k0,j0] = s_val
        if np.linalg.det(S_j_k) < 0:
            #print("negative det", np.linalg.det(S_j_k),
            #      "grid", s_val)
            return -np.inf
        return np.log((np.linalg.det(S_j_k))) * (n-p-1)/2

    logWeights = np.zeros((ngrid,))
    for g in range(ngrid):
        logWeights[g] = log_det_S_j_k(stat_grid[g])

    # normalize logWeights
    logWeights = logWeights - np.max(logWeights)
    # Set extremely small values (< e^-500) to e^-500 for numerical stability
    # logWeights_zero = (logWeights < -500)
    # logWeights[logWeights_zero] = -500

    condlWishart = discrete_family(stat_grid, np.exp(logWeights),
                                   logweights=logWeights)

    neg_interval = condlWishart.equal_tailed_interval(observed=inner_prod,
                                                      alpha=0.1)
    if np.isnan(neg_interval[0]) or np.isnan(neg_interval[1]):
        print("Failed to construct intervals: nan")

    interval = invert_interval(neg_interval)

    pivot = condlWishart.ccdf(theta=0)

    return pivot, interval[0], interval[1]#neg_interval, condlWishart

def edge_inference_scipy(j0, k0, S, n, p, Theta_hat=None, var=None, level=0.9, ngrid=10000):
    # n_total: the total data points in data splitting
    #        : the raw dimension of X in naive
    inner_prod = S[j0,k0]

    # Theta_hat: A low dimensional point estimate of theta
    if Theta_hat is None:
        t_j_k = - inner_prod * n
    else:
        t_j_k = Theta_hat[j0,k0] * n

    #print("t_j_k (center of root finding)", t_j_k)
    #print("theta_hat", Theta_hat[j0,k0])

    S_copy = np.copy(S)

    def log_det_S_j_k(s_val):
        S_j_k = S_copy
        S_j_k[j0,k0] = s_val
        S_j_k[k0,j0] = s_val
        return np.log(np.abs(np.linalg.det(S_j_k))) * (n-p-1)/2
    def det_S_j_k(s_val):
        return np.exp(log_det_S_j_k(s_val))

    def condl_pdf(t,theta=0):
        return det_S_j_k(t) * np.exp(-theta*t)

    def condl_log_pdf(t,theta=0):
        return log_det_S_j_k(t) - theta * t

    def get_pivot(theta0=0, plot=False):
        # print("new run")
        # Normalize the pdf by the maximum over a sparse grid
        def get_pdf_log_normalizer(theta0):
            sparse_grid = np.linspace(-1, 1, num=100)
            sparse_lpdf = np.zeros((100,))
            for g in range(100):
                sparse_lpdf[g] = condl_log_pdf(sparse_grid[g],theta0)
            pdf_log_normalizer = np.max(sparse_lpdf)

            return pdf_log_normalizer

        pdfln = get_pdf_log_normalizer(theta0)
        #print("pdfln", pdfln)
        # Normalized pdf
        def condl_pdf_normalized(t, theta=0):
            return np.exp(log_det_S_j_k(t) - theta*t - pdfln)

        grid_lb = -1.
        grid_ub = 1.
        normalizer = quad(condl_pdf_normalized,
                          grid_lb,
                          grid_ub, args=(theta0,))[0]

        cdf_upper = np.exp(np.log(quad(condl_pdf_normalized, inner_prod, grid_ub,
                         args=(theta0,))[0]) - np.log(normalizer))

        if plot:
            stat_grid = np.zeros((1, ngrid))

            stat_grid[0, :] = np.linspace(grid_lb,
                                          grid_ub,
                                          num=ngrid)
            density = np.zeros((ngrid,))
            for g in range(ngrid):
                density[g] = condl_pdf_normalized(stat_grid[0, g], theta0)

            # print("Max normalized pdf:", np.max(density))
            plt.plot(stat_grid[0, :], density)
        return cdf_upper

    pivot = get_pivot(0)

    # print("bracket", t_j_k - 0.2 * n, t_j_k + 0.2 * n)

    # Construct CI
    margin = (1 - level) / 2
    """root_low = root_scalar(get_pivot_val, bracket=[t_j_k - 0.05 * n, t_j_k + 0.15 * n], args=(margin,),
                           method='bisect')
    root_up = root_scalar(get_pivot_val, bracket=[t_j_k - 0.15 * n, t_j_k + 0.05 * n], args=(1 - margin,),
                          method='bisect')

    return pivot, root_up.root, root_low.root"""
    root_low = find_root(f=get_pivot, y=margin, lb=t_j_k - 0.2 * n, ub=t_j_k + 0.4 * n, tol=1e-4)
    root_up = find_root(f=get_pivot, y=1 - margin, lb=t_j_k - 0.4 * n, ub=t_j_k + 0.2 * n, tol=1e-4)

    #pvtlow = get_pivot(t_j_k - 0.3 * n, plot=True)
    #pvtup = get_pivot(t_j_k + 0.3 * n, plot=True)
    F_low = get_pivot(root_low)
    F_up = get_pivot(root_up)
    #print("p-value at lower root:", F_low)
    #print("p-value at upper root:", F_up)

    return pivot, root_up, root_low

def get_nonzero(active_signs, logic='OR'):
    active_sign_sq = add_diag(active_signs, 0)
    if logic == 'OR':
        nonzero = ((active_sign_sq + active_sign_sq.T) != 0) # OR
    elif logic == 'AND':
        nonzero = ((active_sign_sq * active_sign_sq.T) != 0) # AND
    return nonzero

def conditional_inference(X, nonzero):
    # X is divided by root n, where n is the dimension of X
    # The target of inference is n*Theta (n * prec)
    n,p = X.shape

    # Estimating variances by bootstrap
    # inner_vars = bootstrap_variance(X)
    S_ = X.T @ X
    # theta_h = np.linalg.inv(S_)
    intervals = np.zeros((p,p,2))
    # condlDists = {}
    for i in range(p):
        for j in range(i+1,p):
            if nonzero[i,j]:
                #S_ = X.T @ X
                #theta_h = np.linalg.inv(S_)
                # lcb, ucb are intervals for n * theta
                pivot, lcb, ucb = edge_inference(j0k0=(i,j), S=S_, n=n, p=p, ngrid=10000)
                #pivot, lcb, ucb = edge_inference_scipy(j0=i, k0=j, S=S_,
                 #                                      n=n, p=p, Theta_hat=theta_h)
                # interval = invert_interval(neg_int, scalar=n)
                intervals[i,j,0] = lcb/n
                intervals[i,j,1] = ucb/n
                # print("(", i, ",", j, "): (", lcb/n, ",", ucb/n, ")")
                if ucb/n - lcb/n < 0.01:
                    print("WARNING: SHORT INTERVAL")

    return intervals#, condlDists

def get_coverage(nonzero, intervals, prec, n, p, scale=True):
    # intervals are scaled back to be intervals for theta
    # prec is theta * n, needs to be scaled back
    if scale:
        prec = prec / n

    coverage = np.zeros((p,p))
    for i in range(p):
        for j in range(i+1,p):
            if nonzero[i,j]:
                interval = intervals[i,j,:]
                if prec[i,j] < interval[1] and prec[i,j] > interval[0]:
                    coverage[i,j] = 1
                else:
                    coverage[i,j] = 0
    return coverage

## TODO: TEST
## Naive inference implementations
def naive_inference(X, prec, weights_const=1., true_nonzero=None, logic = 'OR',
                    solve_only=False, continued=False, nonzero_cont=None):
    """
    solve_only: Logical value, determine whether we only want to see
                if this data gives nonzero selection
    continued: If the first run of this function on a data X is solve_only,
                then we set continued to True so that the programs continues
                the selected edges from the previous run
    """
    # Precision matrix is in its original order, not scaled by root n
    # X is also in its original order
    n, p = X.shape
    # rescale X and prec
    X = X / np.sqrt(n)
    prec = prec * n

    if not continued:
        if true_nonzero is not None:
            print("True nonzero used")
            nonzero=true_nonzero
        else:
            print("E estimated")
            nbd_instance = nbd_lasso.gaussian(X, n_scaled=True, weights_const=weights_const)
            active_signs_nonrandom = nbd_instance.fit(perturb=np.zeros((p,p-1)))
            nonzero = get_nonzero(active_signs_nonrandom, logic = logic)

    # If we only need to solve the Lasso
    if solve_only:
        return nonzero

    # If we continue a previous run with a nontrivial selection
    if continued:
        nonzero = nonzero_cont
        assert nonzero.sum() > 0

    # Construct intervals
    if nonzero.sum() > 0:
        # Intervals returned is in its original (unscaled) order
        # intervals, condlDists = conditional_inference(X, nonzero)
        intervals = conditional_inference(X, nonzero)
        # coverage is upper-triangular
        coverage = get_coverage(nonzero, intervals, prec, n, p)
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

## TODO: TEST
def data_splitting(X, prec, weights_const=1., proportion=0.5, logic = 'OR',
                   solve_only=False, continued=False, nonzero_cont=None, subset_cont=None):
    # Precision matrix is in its original order, not scaled by root n
    # X is also in its original order
    n,p = X.shape
    pi_s = proportion
    if not continued:
        subset_select = np.zeros(n, np.bool_)
        subset_select[:int(pi_s * n)] = True
        np.random.shuffle(subset_select)
    else:
        subset_select = subset_cont
    n1 = subset_select.sum()
    n2 = n - n1

    # Rescale X_S and X_NS for numerical stability
    X_S = X[subset_select, :] / np.sqrt(n1)
    X_NS = X[~subset_select, :] / np.sqrt(n2)

    nbd_instance = nbd_lasso.gaussian(X_S, n_scaled=True, weights_const=weights_const)
    active_signs_nonrandom = nbd_instance.fit(perturb=np.zeros((p,p-1)))
    nonzero = get_nonzero(active_signs_nonrandom, logic = logic)
    # print("Data Splitting |E|:", nonzero.sum())

    # If we only need to solve the Lasso
    if solve_only:
        return nonzero, subset_select

    # If we continue a previous run with a nontrivial selection
    if continued:
        nonzero = nonzero_cont
        assert nonzero.sum() > 0

    # Construct intervals
    if nonzero.sum() > 0:
        # Intervals returned is in its original (unscaled) order
        # intervals, condlDists = conditional_inference(X_NS, nonzero=nonzero)
        intervals = conditional_inference(X_NS, nonzero=nonzero)
        # coverage is upper-triangular
        coverage = get_coverage(nonzero, intervals, prec * n2, n2, p)
        interval_len = 0
        nonzero_count = 0
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

def print_nonzero_intervals(nonzero, intervals, prec, X, condlDists):
    # Intervals, prec, X are all in their original scale
    n, p = X.shape
    S = X.T @ X / n

    for i in range(p):
            for j in range(i+1,p):
                if nonzero[i,j]:
                    print("(",i,",",j,")", "selected")
                    print("Theta", "(",i,",",j,")", "interval:", intervals[i,j,:])
                    print("Theta", "(",i,",",j,")", prec[i,j])
                    print("S/n", "(",i,",",j,")", S[i,j])
                    plt.plot(condlDists[(i, j)].sufficient_stat,
                             condlDists[(i, j)].pdf(theta=n*prec[i,j]))
                    plt.title("(" + str(i) + "," + str(j)  + ")")
                    plt.show()


# TODO: TEST
## A function that determines the selected edges based on whether
## the post-selection intervals, no matter valid or not,
## includes zero
def interval_selection(intervals, nonzero):
    p = intervals.shape[0]
    intv_nonzero = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            if i != j and nonzero[i,j]:
                # select an edge if its
                intv_nonzero[i,j] = ( (intervals[i,j,0] * intervals[i,j,1]) > 0 )
    return intv_nonzero


def calculate_F1_score_graph(beta_true, selection):
    # assert is_sym(selection)
    nonzero_true = (beta_true != 0)
    for i in range(nonzero_true.shape[0]):
        # Remove diagonals
        nonzero_true[i,i] = False

    # precision & recall
    if selection.sum() > 0:
        precision = (nonzero_true * selection).sum() / selection.sum()
    else:
        precision = 0
    recall = (nonzero_true * selection).sum() / nonzero_true.sum()

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0


## Parallelization
@timebudget
def run_complex_operations(operation, input, pool):
    pool.map(operation, input)

