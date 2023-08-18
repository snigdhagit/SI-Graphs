from __future__ import print_function

from typing import NamedTuple

import numpy as np

import regreg.api as rr

from multiprocessing import Pool

from functools import partial

from scipy.stats import norm

from .query import gaussian_query

from .randomization import randomization

from .Utils.base import (restricted_estimator,
                         _compute_hessian)

from .nbd_helpers import *

from .approx_reference_nbd import *


#### High dimensional version
#### - parametric covariance
#### - Gaussian randomization

class QuerySpec(NamedTuple):
    # the covariance(s) of randomization(s)
    cov_rands: list
    prec_rands: list

    # list of constraints
    linear_parts: list
    offsets: list

    # list of ridge terms
    ridge_terms: list

    # observed values
    nonzero: np.ndarray
    active: np.ndarray
    observed_subgrad: np.ndarray
    observed_soln: np.ndarray

class nbd_lasso(object):

    def __init__(self,
                 X,
                 loglike,
                 weights,
                 ridge_terms,
                 randomizer):

        n = X.shape[0]
        self.X_n = X / np.sqrt(n) # Scaled version of X

        self.nfeature = X.shape[1]
        if np.asarray(weights).shape == ():
            weights = np.ones((self.nfeature,self.nfeature - 1)) * weights
            #print(weights.shape)
            #print(weights)
        self.weights = weights
        self.loglike = loglike
        # print(weights.shape)
        # print("weights[5]:",weights[5])

        # ridge parameter
        self.ridge_terms = ridge_terms

        self.penalty = []
        for i in range(self.nfeature):
            self.penalty.append(rr.weighted_l1norm(self.weights[i], lagrange=1.))

        self._initial_omega = None

        # gaussian randomization
        self.randomizer = randomizer

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):
        """
        Fit the randomized lasso using `regreg`.
        Parameters
        ----------
        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.
        Returns
        -------
        signs : np.float
             Support and non-zero signs of randomized lasso solution.
        """

        p = self.nfeature

        # Two matrices of dimension p x p-1
        (self.observed_soln,
         self.observed_subgrad) = self._solve_randomized_problem(perturb=perturb,
                                                                 solve_args=solve_args)

        active_signs = np.sign(self.observed_soln)
        # Nonzero flag
        self._active = active_signs != 0
        # Determine selection with OR logic
        self.nonzero = get_nonzero(self._active, logic='OR')

        self.cov_rands = []
        self.prec_rands = []
        self.linear_parts = []
        self.offsets = []

        for i in range(p):
            self.cov_rands.append(self.randomizer[i].cov_prec[0] * np.eye(p - 1))
            self.prec_rands.append(self.randomizer[i].cov_prec[1] * np.eye(p - 1))
            sum_nonzero_i = self._active[i,:].sum()
            if sum_nonzero_i > 1:
                self.linear_parts.append(-np.diag(active_signs[i,self._active[i,:]]))
                self.offsets.append(np.zeros(sum_nonzero_i))
            elif sum_nonzero_i == 1:
                self.linear_parts.append(-active_signs[i, self._active[i, :]])
                self.offsets.append(0)
            else:
                self.linear_parts.append(None)
                self.offsets.append(None)
            """print("i:", i)
            print("|E_i|:", sum_nonzero_i)
            print("linear:", -np.diag(active_signs[i,self._active[i,:]]))
            print("offset:", np.zeros(self._active[i,:].sum()))"""
            if sum_nonzero_i != 0:
                if not np.all(self.linear_parts[i].dot(self.observed_soln[i,self._active[i,:]])
                              - self.offsets[i] <= 0):
                    raise ValueError(str(i) + 'th constraint not satisfied')

        return active_signs

    def _solve_randomized_problem(self,
                                  perturb=None,
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if supplied
        if perturb is not None:
            #print("Custom perturbation")
            assert perturb.shape == (self.nfeature, self.nfeature-1)
            self._initial_omega = perturb
        else:
            #print("Sampled perturbation")
            self._initial_omega = np.zeros((self.nfeature, self.nfeature-1))
            for i in range(self.nfeature):
                self._initial_omega[i] = self.randomizer[i].sample()

        quad = []
        for i in range(self.nfeature):
            quad_i = rr.identity_quadratic(self.ridge_terms[i],
                                         0,
                                         -self._initial_omega[i],
                                         0)
            quad.append(quad_i)

        observed_soln = np.zeros((self.nfeature, self.nfeature-1))
        observed_subgrad = np.zeros((self.nfeature, self.nfeature - 1))
        for i in range(self.nfeature):
            problem_i = rr.simple_problem(self.loglike[i], self.penalty[i])
            observed_soln[i] = problem_i.solve(quad[i], **solve_args)
            observed_subgrad[i] = -(self.loglike[i].smooth_objective(observed_soln[i],
                                                                  'grad') +
                                    quad[i].objective(observed_soln[i], 'grad'))

        return observed_soln, observed_subgrad

    @property
    def specification(self):
        return QuerySpec(cov_rands=self.cov_rands,
                         prec_rands=self.prec_rands,
                         linear_parts=self.linear_parts,  # linear_part o < offset
                         offsets=self.offsets,
                         ridge_terms=self.ridge_terms,
                         nonzero=self.nonzero,
                         active=self._active,
                         observed_subgrad=self.observed_subgrad,
                         observed_soln=self.observed_soln)

    def inference(self, level=0.9, parallel=True):

        query_spec = self.specification
        nonzero = query_spec.nonzero

        # X is divided by root n, where n is the dimension of X
        # The target of inference is n*Theta (n * prec)
        X_n = self.X_n
        n, p = X_n.shape

        S_ = X_n.T @ X_n
        intervals = np.zeros((p, p, 2))

        if parallel:
            task_idx = []
            for i in range(p):
                for j in range(i + 1, p):
                    if nonzero[i, j]:
                        task_idx.append((i, j))
            with Pool() as pool:
                results = pool.map(partial(approx_inference, X_n=X_n, query_spec=query_spec,
                                           n=n, p=p, ngrid=10000, ncoarse=50, level=level),
                                   task_idx)
            for t in range(len(task_idx)):
                pivot, lcb, ucb = results[t]
                i = task_idx[t][0]
                j = task_idx[t][1]
                intervals[i, j, 0] = lcb / n
                intervals[i, j, 1] = ucb / n
                # print("(", i, ",", j, "): (", lcb/n, ",", ucb/n, ")")
                if ucb / n - lcb / n < 0.01:
                    print("WARNING: SHORT INTERVAL")
        else:
            for i in range(p):
                for j in range(i+1, p):
                    if nonzero[i, j]:
                        print("Inference for", i, ",", j)
                        pivot, lcb, ucb = approx_inference(query_spec=query_spec,
                                                           j0k0=(i,j), X_n=X_n, n=n, p=p,
                                                           ngrid=10000, ncoarse=50, level=level)
                        intervals[i, j, 0] = lcb / n
                        intervals[i, j, 1] = ucb / n
                        # print("(", i, ",", j, "): (", lcb/n, ",", ucb/n, ")")
                        if ucb / n - lcb / n < 0.01:
                            print("WARNING: SHORT INTERVAL")

        return intervals  # , condlDists

    @staticmethod
    def gaussian(X,
                 alpha=0.1,
                 feature_weights=None,
                 weights_const=1.,
                 quadratic=None,
                 ridge_terms=None,
                 randomizer_scale=None,
                 nonrandomized=False,
                 n_scaled=True):
        r"""
        Squared-error LASSO with feature weights.
        Objective function is (before randomization)
        .. math::
            \beta \mapsto \frac{1}{2} \|Y-X\beta\|^2_2 + \sum_{i=1}^p \lambda_i |\beta_i|
        where $\lambda$ is `feature_weights`. The ridge term
        is determined by the Hessian by default,
        as is the randomizer scale.
        Parameters
        ----------
        X : ndarray
            Shape (n,p) -- the design matrix.
        Y : ndarray
            Shape (n,) -- the response.
        feature_weights: [float, sequence]
            Penalty weights. An intercept, or other unpenalized
            features are handled by setting those entries of
            `feature_weights` to 0. If `feature_weights` is
            a float, then all parameters are penalized equally.
        sigma : float (optional)
            Noise variance. Set to 1 if `covariance_estimator` is not None.
            This scales the loglikelihood by `sigma**(-2)`.
        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            An optional quadratic term to be added to the objective.
            Can also be a linear term by setting quadratic
            coefficient to 0.
        ridge_term : float
            How big a ridge term to add?
        randomizer_scale : float
            Scale for IID components of randomizer.
        Returns
        -------
        L : `selection.randomized.lasso.lasso`
        """

        n, p = X.shape
        loglike = []
        for i in range(p):
            loglike_i = rr.glm.gaussian(np.delete(X, i, axis=1),
                                        X[:,i],
                                        coef=1.,
                                        quadratic=quadratic)
            loglike.append(loglike_i)

        """if ridge_term is None:
            ridge_term = 0."""

        if ridge_terms is None:
            ridge_terms = []
            for i in range(p):
                X_i = X[:, list(j for j in range(p) if j != i)]
                mean_diag = np.mean((X_i ** 2).sum(0))
                # Should be approximately 1 for standardized X
                ridge_term_i = np.sqrt(mean_diag) / (np.sqrt(n - 1))
                ridge_terms.append(ridge_term_i)
        elif np.asarray(ridge_terms).shape == ():
            ridge_const = ridge_terms
            ridge_terms = []
            for i in range(p):
                ridge_terms.append(ridge_const)


        if feature_weights is None:
            def Phi_tilde_inv(a):
                return -norm.ppf(a)
            feature_weights = []
            for i in range(p):
                sigma_i = np.sqrt(np.sum(X[:,i]**2) / n)
                # print("sigma_i:", sigma_i)
                if n_scaled:
                    # X with root n scaling
                    weight_sclar = 2 * weights_const * sigma_i * Phi_tilde_inv(alpha / (2 * p ** 2))
                else:
                    # Pre root n scaling:
                    weight_sclar = 2 * weights_const * np.sqrt(n) * sigma_i * Phi_tilde_inv(alpha / (2 * p ** 2))

                feature_weights_i = np.ones(p-1) * weight_sclar
                feature_weights.append(feature_weights_i)

        if randomizer_scale is None:
            randomizer_scale = 1. # 0.5?

        randomizer = []
        for i in range(p):
            if n_scaled:
                randomizer_scale_i = randomizer_scale * np.std(X[:, i], ddof=1)
            else:
                randomizer_scale_i = randomizer_scale * np.std(X[:, i], ddof=1) * np.sqrt(n)

            randomizer.append(randomization.isotropic_gaussian((p - 1,), randomizer_scale_i))

        return nbd_lasso(X=X,
                         loglike=loglike,
                         weights=np.asarray(feature_weights),
                         ridge_terms=ridge_terms,
                         randomizer=randomizer)