from __future__ import print_function

import numpy as np

import regreg.api as rr

from .query import gaussian_query

from .randomization import randomization

from .Utils.base import (restricted_estimator,
                         _compute_hessian)


#### High dimensional version
#### - parametric covariance
#### - Gaussian randomization

class lasso(gaussian_query):
    r"""
    A class for the randomized LASSO for post-selection inference.
    The problem solved is
    .. math::
        \text{minimize}_{\beta} \ell(\beta) +
            \sum_{i=1}^p \lambda_i |\beta_i\| - \omega^T\beta + \frac{\epsilon}{2} \|\beta\|^2_2
    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty. Each static method
    forms $\ell$ as well as the $\ell_1$ penalty. The generic class
    forms the remaining two terms in the objective.
    """

    def __init__(self,
                 loglike,
                 feature_weights,
                 ridge_term,
                 randomizer,
                 perturb=None):
        r"""
        Create a new post-selection object for the LASSO problem
        Parameters
        ----------
        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.
        feature_weights : np.ndarray
            Feature weights for L-1 penalty. If a float,
            it is brodcast to all features.
        ridge_term : float
            How big a ridge term to add?
        randomizer : object
            Randomizer -- contains representation of randomization density.
        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.
        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(feature_weights).shape == ():
            feature_weights = np.ones(loglike.shape) * feature_weights
        self.feature_weights = np.asarray(feature_weights)

        self.ridge_term = ridge_term
        self.penalty = rr.weighted_l1norm(self.feature_weights, lagrange=1.)
        self._initial_omega = perturb  # random perturbation

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

        (self.observed_soln,
         self.observed_subgrad) = self._solve_randomized_problem(
            perturb=perturb,
            solve_args=solve_args)

        active_signs = np.sign(self.observed_soln)
        active = self._active = active_signs != 0

        self._lagrange = self.penalty.weights
        unpenalized = self._lagrange == 0

        active *= ~unpenalized

        self._overall = overall = (active + unpenalized) > 0
        self._inactive = inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()

        # don't release sign of unpenalized variables
        _active_signs[unpenalized] = np.nan
        ordered_variables = list((tuple(np.nonzero(active)[0]) +
                                  tuple(np.nonzero(unpenalized)[0])))
        self.selection_variable = {'sign': _active_signs,
                                   'variables': ordered_variables}

        # initial state for opt variables

        initial_scalings = np.fabs(self.observed_soln[active])
        initial_unpenalized = self.observed_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized])

        _beta_unpenalized = restricted_estimator(self.loglike,
                                                 self._overall,
                                                 solve_args=solve_args)


        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar


        num_opt_var = self.observed_opt_state.shape[0]

        _hessian, _hessian_active, _hessian_unpen = _compute_hessian(self.loglike,
                                                                     beta_bar,
                                                                     active,
                                                                     unpenalized)

        opt_linear = np.zeros((p, num_opt_var))
        _score_linear_term = np.zeros((p, num_opt_var))

        _score_linear_term = -np.hstack([_hessian_active, _hessian_unpen])


        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p,
                                                          j,
                                                          active_signs[j])
                                      for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = (_hessian_active * active_signs[None, active]
                            + self.ridge_term * active_directions)

        opt_linear[:, scaling_slice] = _opt_hessian

        unpenalized_slice = slice(active.sum(), num_opt_var)
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for
                                           j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            opt_linear[:, unpenalized_slice] = (_hessian_unpen
                                                + self.ridge_term *
                                                unpenalized_directions)

        self.opt_linear = opt_linear


        self._setup = True
        A_scaling = -np.identity(num_opt_var)
        b_scaling = np.zeros(num_opt_var)

        self._unscaled_cov_score = _hessian

        self.num_opt_var = num_opt_var

        self._setup_sampler_data = (A_scaling[:active.sum()],
                                    b_scaling[:active.sum()],
                                    opt_linear,
                                    self.observed_subgrad)

        return active_signs

    def setup_inference(self,
                        dispersion):

        if self.num_opt_var > 0:
            self._setup_sampler(*self._setup_sampler_data,
                                dispersion=dispersion)

    def _solve_randomized_problem(self,
                                  perturb=None,
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term,
                                     0,
                                     -self._initial_omega,
                                     0)

        problem = rr.simple_problem(self.loglike, self.penalty)

        observed_soln = problem.solve(quad, **solve_args)
        observed_subgrad = -(self.loglike.smooth_objective(observed_soln,
                                                           'grad') +
                             quad.objective(observed_soln, 'grad'))

        return observed_soln, observed_subgrad

    @staticmethod
    def gaussian(X,
                 Y,
                 feature_weights,
                 sigma=1.,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scale=None):
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

        loglike = rr.glm.gaussian(X,
                                  Y,
                                  coef=1. / sigma ** 2,
                                  quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))

        if ridge_term is None:
            ridge_term = np.sqrt(mean_diag) / (np.sqrt(n - 1) * sigma ** 2)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y, ddof=1)

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return lasso(loglike,
                     np.asarray(feature_weights) / sigma ** 2,
                     ridge_term,
                     randomizer)
