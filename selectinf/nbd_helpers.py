import numpy as np
import random
import warnings

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

def get_nonzero(active_signs, logic='OR'):
    active_sign_sq = add_diag(active_signs, 0)
    if logic == 'AND':
        nonzero = ((active_sign_sq * active_sign_sq.T) != 0)  # AND
    else:
        nonzero = ((active_sign_sq + active_sign_sq.T) != 0)  # OR
    return nonzero
