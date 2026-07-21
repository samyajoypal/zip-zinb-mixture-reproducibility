"""Analytical likelihood pieces for the ZIP--ZINB project.

This module is intentionally small and explicit.  It is the new, independent
analytical core; the copied legacy source is kept under ``legacy_source/`` and
is not imported here.

Implemented in this first pass:

- Poisson log-PMF and score with respect to mean-regression coefficients.
- NB2 log-PMF and score with respect to mean-regression coefficients and
  log-dispersion.
- ZIP and ZINB component log-likelihoods and analytical scores.
- Full ZIP--ZINB observed log-likelihood and analytical gradient under the
  unconstrained parameterization:

  ``[logit_rho, beta_zip, gamma_zip, beta_zinb, gamma_zinb, log_alpha]``.

The Hessian/information implementation will build on the same per-observation
component scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, gammaln, logsumexp, digamma, polygamma


EPS = 1e-12


@dataclass(frozen=True)
class LinkDerivatives:
    """Mean and first two derivatives with respect to eta."""

    mu: np.ndarray
    d1: np.ndarray
    d2: np.ndarray


def _as_1d(y: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(y, dtype=float).reshape(-1)


def add_intercept(x: np.ndarray | list[float]) -> np.ndarray:
    """Return a 2D design matrix with a leading intercept column."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.column_stack([np.ones(arr.shape[0]), arr])


def mean_link(eta: np.ndarray, link: str) -> LinkDerivatives:
    """Inverse mean link and derivatives.

    For the identity link we do *not* clip. The analytical formulation requires
    the caller/optimizer to keep the linear predictor positive.
    """

    eta = np.asarray(eta, dtype=float)
    if link == "log":
        mu = np.exp(eta)
        return LinkDerivatives(mu=mu, d1=mu, d2=mu)
    if link == "identity":
        return LinkDerivatives(mu=eta, d1=np.ones_like(eta), d2=np.zeros_like(eta))
    raise ValueError(f"Unsupported link: {link!r}")


def poisson_logpmf(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Poisson log-PMF."""

    return y * np.log(mu) - mu - gammaln(y + 1.0)


def poisson_beta_score(y: np.ndarray, X: np.ndarray, beta: np.ndarray, link: str) -> tuple[np.ndarray, np.ndarray]:
    """Poisson log-PMF and score wrt beta, row by row."""

    eta = X @ beta
    ld = mean_link(eta, link)
    if np.any(ld.mu <= 0):
        raise FloatingPointError("Mean must be positive for analytical likelihood.")
    logp = poisson_logpmf(y, ld.mu)
    score_eta = (y / ld.mu - 1.0) * ld.d1
    return logp, score_eta[:, None] * X


def nb2_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """NB2 log-PMF with Var(Y)=mu+alpha*mu^2."""

    if alpha <= 0:
        raise FloatingPointError("alpha must be positive.")
    if np.any(mu <= 0):
        raise FloatingPointError("mu must be positive.")
    r = 1.0 / alpha
    log_1amu = np.log1p(alpha * mu)
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        - r * log_1amu
        + y * (np.log(alpha) + np.log(mu) - log_1amu)
    )


def nb2_beta_logalpha_score(
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    log_alpha: float,
    link: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NB2 log-PMF and scores wrt beta and log(alpha), row by row."""

    alpha = float(np.exp(log_alpha))
    eta = X @ beta
    ld = mean_link(eta, link)
    if np.any(ld.mu <= 0):
        raise FloatingPointError("Mean must be positive for analytical likelihood.")

    mu = ld.mu
    logp = nb2_logpmf(y, mu, alpha)

    # d log f / d mu for NB2
    dlog_dmu = y / mu - (1.0 + alpha * y) / (1.0 + alpha * mu)
    beta_score = (dlog_dmu * ld.d1)[:, None] * X

    # d log f / d alpha, then chain to log_alpha.
    # log f = lgamma(y+1/a)-lgamma(1/a)-lgamma(y+1)
    #         -(1/a)log(1+a*mu) + y[log(a)+log(mu)-log(1+a*mu)]
    log_1amu = np.log1p(alpha * mu)
    dlog_dalpha = (
        (digamma(1.0 / alpha) - digamma(y + 1.0 / alpha) + log_1amu) / alpha**2
        - mu / (alpha * (1.0 + alpha * mu))
        + y / alpha
        - y * mu / (1.0 + alpha * mu)
    )
    logalpha_score = alpha * dlog_dalpha

    return logp, beta_score, logalpha_score


def zip_loglik_score(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    link: str,
) -> tuple[np.ndarray, np.ndarray]:
    """ZIP component log-likelihood and analytical row-wise score."""

    y = _as_1d(y)
    pois_logp, pois_score = poisson_beta_score(y, X, beta, link)
    pi = np.clip(expit(Z @ gamma), EPS, 1.0 - EPS)
    is_zero = y == 0

    log_count_weight = np.log1p(-pi)
    log_struct_weight = np.log(pi)

    logp = np.empty_like(y, dtype=float)
    logp[~is_zero] = log_count_weight[~is_zero] + pois_logp[~is_zero]
    logp[is_zero] = np.logaddexp(
        log_struct_weight[is_zero],
        log_count_weight[is_zero] + pois_logp[is_zero],
    )

    # Posterior probability of structural zero inside the ZIP component.
    struct_resp = np.zeros_like(y, dtype=float)
    struct_resp[is_zero] = np.exp(log_struct_weight[is_zero] - logp[is_zero])
    count_resp = 1.0 - struct_resp

    beta_score = count_resp[:, None] * pois_score
    gamma_score = (struct_resp - pi)[:, None] * Z
    return logp, np.column_stack([beta_score, gamma_score])


def zinb_loglik_score(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    log_alpha: float,
    link: str,
) -> tuple[np.ndarray, np.ndarray]:
    """ZINB/NB2 component log-likelihood and analytical row-wise score."""

    y = _as_1d(y)
    nb_logp, nb_beta_score, nb_logalpha_score = nb2_beta_logalpha_score(
        y, X, beta, log_alpha, link
    )
    omega = np.clip(expit(Z @ gamma), EPS, 1.0 - EPS)
    is_zero = y == 0

    log_count_weight = np.log1p(-omega)
    log_struct_weight = np.log(omega)

    logp = np.empty_like(y, dtype=float)
    logp[~is_zero] = log_count_weight[~is_zero] + nb_logp[~is_zero]
    logp[is_zero] = np.logaddexp(
        log_struct_weight[is_zero],
        log_count_weight[is_zero] + nb_logp[is_zero],
    )

    struct_resp = np.zeros_like(y, dtype=float)
    struct_resp[is_zero] = np.exp(log_struct_weight[is_zero] - logp[is_zero])
    count_resp = 1.0 - struct_resp

    beta_score = count_resp[:, None] * nb_beta_score
    gamma_score = (struct_resp - omega)[:, None] * Z
    logalpha_score = count_resp * nb_logalpha_score
    return logp, np.column_stack([beta_score, gamma_score, logalpha_score])


def poisson_loglike_gradient(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray]:
    """Poisson observed log-likelihood and analytical gradient."""

    y = _as_1d(y)
    beta = np.asarray(params, dtype=float)
    logp, beta_score = poisson_beta_score(y, X, beta, link)
    return float(logp.sum()), beta_score.sum(axis=0)


def nb2_loglike_gradient(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray]:
    """NB2 observed log-likelihood and analytical gradient."""

    y = _as_1d(y)
    params = np.asarray(params, dtype=float)
    beta = params[:-1]
    log_alpha = float(params[-1])
    logp, beta_score, logalpha_score = nb2_beta_logalpha_score(
        y, X, beta, log_alpha, link
    )
    return float(logp.sum()), np.r_[beta_score.sum(axis=0), logalpha_score.sum()]


def zip_loglike_gradient(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray]:
    """ZIP observed log-likelihood and analytical gradient."""

    p = X.shape[1]
    beta = np.asarray(params[:p], dtype=float)
    gamma = np.asarray(params[p:], dtype=float)
    logp, score = zip_loglik_score(y, X, Z, beta, gamma, link)
    return float(logp.sum()), score.sum(axis=0)


def zinb_loglike_gradient(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray]:
    """ZINB/NB2 observed log-likelihood and analytical gradient."""

    p, q = X.shape[1], Z.shape[1]
    beta = np.asarray(params[:p], dtype=float)
    gamma = np.asarray(params[p : p + q], dtype=float)
    log_alpha = float(params[p + q])
    logp, score = zinb_loglik_score(y, X, Z, beta, gamma, log_alpha, link)
    return float(logp.sum()), score.sum(axis=0)


def pnb_loglike_gradient(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray]:
    """Poisson+NB2 mixture observed log-likelihood and gradient.

    Parameter order:
    ``[logit_rho, beta_p, beta_nb, log_alpha]``.
    """

    y = _as_1d(y)
    p = X.shape[1]
    params = np.asarray(params, dtype=float)
    logit_rho = float(params[0])
    beta_p = params[1 : 1 + p]
    beta_nb = params[1 + p : 1 + 2 * p]
    log_alpha = float(params[-1])
    rho = float(expit(logit_rho))

    log_pois, score_pois = poisson_beta_score(y, X, beta_p, link)
    log_nb, score_nb_beta, score_nb_logalpha = nb2_beta_logalpha_score(
        y, X, beta_nb, log_alpha, link
    )
    two_cols = np.column_stack([
        np.log(rho) + log_pois,
        np.log1p(-rho) + log_nb,
    ])
    log_mix = logsumexp(two_cols, axis=1)
    kappa = np.exp(two_cols[:, 0] - log_mix)

    ll = float(log_mix.sum())
    grad = np.r_[
        (kappa - rho).sum(),
        (kappa[:, None] * score_pois).sum(axis=0),
        ((1.0 - kappa)[:, None] * score_nb_beta).sum(axis=0),
        ((1.0 - kappa) * score_nb_logalpha).sum(),
    ]
    return ll, grad


def zpnb_loglike_gradient(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray]:
    """Shared-inflation zero-inflated Poisson+NB2 mixture gradient.

    Parameter order:
    ``[logit_rho, beta_p, beta_nb, gamma, log_alpha]``.
    """

    y = _as_1d(y)
    p, q = X.shape[1], Z.shape[1]
    params = np.asarray(params, dtype=float)
    logit_rho = float(params[0])
    beta_p = params[1 : 1 + p]
    beta_nb = params[1 + p : 1 + 2 * p]
    gamma = params[1 + 2 * p : 1 + 2 * p + q]
    log_alpha = float(params[-1])
    rho = float(expit(logit_rho))

    log_pois, score_pois = poisson_beta_score(y, X, beta_p, link)
    log_nb, score_nb_beta, score_nb_logalpha = nb2_beta_logalpha_score(
        y, X, beta_nb, log_alpha, link
    )
    two_cols = np.column_stack([
        np.log(rho) + log_pois,
        np.log1p(-rho) + log_nb,
    ])
    log_count_mix = logsumexp(two_cols, axis=1)
    kappa_count = np.exp(two_cols[:, 0] - log_count_mix)

    pi = np.clip(expit(Z @ gamma), EPS, 1.0 - EPS)
    is_zero = y == 0
    log_count_weight = np.log1p(-pi)
    log_struct_weight = np.log(pi)

    logp = np.empty_like(y, dtype=float)
    logp[~is_zero] = log_count_weight[~is_zero] + log_count_mix[~is_zero]
    logp[is_zero] = np.logaddexp(
        log_struct_weight[is_zero],
        log_count_weight[is_zero] + log_count_mix[is_zero],
    )

    struct_resp = np.zeros_like(y, dtype=float)
    struct_resp[is_zero] = np.exp(log_struct_weight[is_zero] - logp[is_zero])
    count_resp = 1.0 - struct_resp

    grad = np.r_[
        (count_resp * (kappa_count - rho)).sum(),
        (count_resp[:, None] * kappa_count[:, None] * score_pois).sum(axis=0),
        (
            count_resp[:, None]
            * (1.0 - kappa_count)[:, None]
            * score_nb_beta
        ).sum(axis=0),
        ((struct_resp - pi)[:, None] * Z).sum(axis=0),
        (count_resp * (1.0 - kappa_count) * score_nb_logalpha).sum(),
    ]
    return float(logp.sum()), grad


def unpack_full_params(params: np.ndarray, p: int, q: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Unpack full ZIP--ZINB transformed parameter vector."""

    params = np.asarray(params, dtype=float)
    expected = 1 + p + q + p + q + 1
    if params.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {params.size}.")

    idx = 0
    logit_rho = float(params[idx])
    idx += 1
    beta_zip = params[idx : idx + p]
    idx += p
    gamma_zip = params[idx : idx + q]
    idx += q
    beta_zinb = params[idx : idx + p]
    idx += p
    gamma_zinb = params[idx : idx + q]
    idx += q
    log_alpha = float(params[idx])
    return logit_rho, beta_zip, gamma_zip, beta_zinb, gamma_zinb, log_alpha


def full_zip_zinb_loglike_gradient(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str = "log",
) -> tuple[float, np.ndarray]:
    """Observed log-likelihood and analytical gradient for full ZIP--ZINB."""

    y = _as_1d(y)
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    p, q = X.shape[1], Z.shape[1]
    logit_rho, beta_z, gamma_z, beta_n, gamma_n, log_alpha = unpack_full_params(
        params, p, q
    )

    rho = float(expit(logit_rho))
    log_zip, score_zip = zip_loglik_score(y, X, Z, beta_z, gamma_z, link)
    log_zinb, score_zinb = zinb_loglik_score(y, X, Z, beta_n, gamma_n, log_alpha, link)

    two_cols = np.column_stack([
        np.log(rho) + log_zip,
        np.log1p(-rho) + log_zinb,
    ])
    log_mix = logsumexp(two_cols, axis=1)
    kappa = np.exp(two_cols[:, 0] - log_mix)

    ll = float(log_mix.sum())

    grad_logit_rho = np.array([(kappa - rho).sum()])
    grad_zip = (kappa[:, None] * score_zip).sum(axis=0)
    grad_zinb = ((1.0 - kappa)[:, None] * score_zinb).sum(axis=0)
    grad = np.r_[grad_logit_rho, grad_zip, grad_zinb]
    return ll, grad


def _row_outer(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Row-wise outer products."""

    a = np.asarray(a, dtype=float)
    b = a if b is None else np.asarray(b, dtype=float)
    return np.einsum("ni,nj->nij", a, b)


def _component_mixture_log_score_hessian(
    log_components: np.ndarray,
    scores: np.ndarray,
    hessians: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-sum-exp mixture with analytical score/Hessian.

    Parameters
    ----------
    log_components:
        Array with shape ``(n, m)`` containing component log contributions.
    scores:
        Array with shape ``(n, m, k)`` containing component scores.
    hessians:
        Array with shape ``(n, m, k, k)`` containing component Hessians.

    Returns
    -------
    logp, score, hessian
        Per-observation log likelihood, score, and Hessian.
    """

    logp = logsumexp(log_components, axis=1)
    weights = np.exp(log_components - logp[:, None])
    score = np.einsum("nm,nmk->nk", weights, scores)
    second_raw = hessians + np.einsum("nmk,nml->nmkl", scores, scores)
    hessian = np.einsum("nm,nmkl->nkl", weights, second_raw) - _row_outer(score)
    return logp, score, hessian


def poisson_log_score_hessian(
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    link: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poisson log-PMF, score, and Hessian wrt beta, row by row."""

    y = _as_1d(y)
    eta = X @ beta
    ld = mean_link(eta, link)
    if np.any(ld.mu <= 0):
        raise FloatingPointError("Mean must be positive for analytical likelihood.")
    logp = poisson_logpmf(y, ld.mu)
    score_eta = (y / ld.mu - 1.0) * ld.d1
    hess_eta = y * (ld.d2 / ld.mu - (ld.d1**2) / (ld.mu**2)) - ld.d2
    score = score_eta[:, None] * X
    hessian = hess_eta[:, None, None] * _row_outer(X)
    return logp, score, hessian


def nb2_log_score_hessian(
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    log_alpha: float,
    link: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NB2 log-PMF, score, and Hessian wrt beta/log-alpha, row by row."""

    y = _as_1d(y)
    alpha = float(np.exp(log_alpha))
    eta = X @ beta
    ld = mean_link(eta, link)
    if np.any(ld.mu <= 0):
        raise FloatingPointError("Mean must be positive for analytical likelihood.")
    mu = ld.mu
    p = X.shape[1]
    logp = nb2_logpmf(y, mu, alpha)

    D = 1.0 + alpha * mu
    A = y / mu - (1.0 + alpha * y) / D
    B = -y / (mu**2) + alpha * (1.0 + alpha * y) / (D**2)
    score_eta = A * ld.d1
    hess_eta = B * (ld.d1**2) + A * ld.d2

    score_beta = score_eta[:, None] * X
    hess_beta_beta = hess_eta[:, None, None] * _row_outer(X)

    log_1amu = np.log1p(alpha * mu)
    T = digamma(1.0 / alpha) - digamma(y + 1.0 / alpha) + log_1amu
    dlog_dalpha = (
        T / alpha**2
        - mu / (alpha * D)
        + y / alpha
        - y * mu / D
    )
    score_psi = alpha * dlog_dalpha

    # Cross derivative d^2 log f / d eta d log(alpha).
    cross_eta_psi = ld.d1 * alpha * (mu - y) / (D**2)
    hess_beta_psi = cross_eta_psi[:, None] * X

    T_prime = (
        (polygamma(1, y + 1.0 / alpha) - polygamma(1, 1.0 / alpha))
        / alpha**2
        + mu / D
    )
    d2log_dalpha2 = (
        T_prime / alpha**2
        - 2.0 * T / alpha**3
        + mu * (1.0 + 2.0 * alpha * mu) / (alpha**2 * D**2)
        - y / alpha**2
        + y * mu**2 / D**2
    )
    hess_psi_psi = alpha * dlog_dalpha + alpha**2 * d2log_dalpha2

    score = np.column_stack([score_beta, score_psi])
    hessian = np.zeros((len(y), p + 1, p + 1), dtype=float)
    hessian[:, :p, :p] = hess_beta_beta
    hessian[:, :p, p] = hess_beta_psi
    hessian[:, p, :p] = hess_beta_psi
    hessian[:, p, p] = hess_psi_psi
    return logp, score, hessian


def _zero_inflated_log_score_hessian(
    y: np.ndarray,
    Z: np.ndarray,
    count_logp: np.ndarray,
    count_score: np.ndarray,
    count_hessian: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic zero-inflation wrapper for an arbitrary count component."""

    y = _as_1d(y)
    n, k_count = count_score.shape
    q = Z.shape[1]
    k = k_count + q
    pi = np.clip(expit(Z @ gamma), EPS, 1.0 - EPS)
    is_zero = y == 0
    zz = _row_outer(Z)
    gamma_hess = -(pi * (1.0 - pi))[:, None, None] * zz

    log_components = np.full((n, 2), -np.inf, dtype=float)
    scores = np.zeros((n, 2, k), dtype=float)
    hessians = np.zeros((n, 2, k, k), dtype=float)

    # Structural-zero component, available only when y = 0.
    log_components[is_zero, 0] = np.log(pi[is_zero])
    scores[:, 0, k_count:] = (1.0 - pi)[:, None] * Z
    hessians[:, 0, k_count:, k_count:] = gamma_hess

    # Count component.
    log_components[:, 1] = np.log1p(-pi) + count_logp
    scores[:, 1, :k_count] = count_score
    scores[:, 1, k_count:] = -pi[:, None] * Z
    hessians[:, 1, :k_count, :k_count] = count_hessian
    hessians[:, 1, k_count:, k_count:] = gamma_hess

    return _component_mixture_log_score_hessian(log_components, scores, hessians)


def zip_log_score_hessian(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    link: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ZIP log-likelihood, score, and Hessian, row by row."""

    count_logp, count_score, count_hessian = poisson_log_score_hessian(
        y, X, beta, link
    )
    return _zero_inflated_log_score_hessian(
        y, Z, count_logp, count_score, count_hessian, gamma
    )


def zinb_log_score_hessian(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    log_alpha: float,
    link: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ZINB/NB2 log-likelihood, score, and Hessian, row by row."""

    count_logp, count_score, count_hessian = nb2_log_score_hessian(
        y, X, beta, log_alpha, link
    )
    logp, score, hessian = _zero_inflated_log_score_hessian(
        y, Z, count_logp, count_score, count_hessian, gamma
    )
    # Wrapper order is [beta, log_alpha, gamma]; public ZINB order is
    # [beta, gamma, log_alpha].
    p = X.shape[1]
    q = Z.shape[1]
    perm = list(range(p)) + list(range(p + 1, p + 1 + q)) + [p]
    return logp, score[:, perm], hessian[:, perm][:, :, perm]


def poisson_loglike_gradient_hessian(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    logp, score, hessian = poisson_log_score_hessian(y, X, params, link)
    return float(logp.sum()), score.sum(axis=0), hessian.sum(axis=0)


def nb2_loglike_gradient_hessian(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=float)
    logp, score, hessian = nb2_log_score_hessian(
        y, X, params[:-1], float(params[-1]), link
    )
    return float(logp.sum()), score.sum(axis=0), hessian.sum(axis=0)


def zip_loglike_gradient_hessian(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    p = X.shape[1]
    logp, score, hessian = zip_log_score_hessian(
        y, X, Z, params[:p], params[p:], link
    )
    return float(logp.sum()), score.sum(axis=0), hessian.sum(axis=0)


def zinb_loglike_gradient_hessian(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    p, q = X.shape[1], Z.shape[1]
    logp, score, hessian = zinb_log_score_hessian(
        y, X, Z, params[:p], params[p : p + q], float(params[p + q]), link
    )
    return float(logp.sum()), score.sum(axis=0), hessian.sum(axis=0)


def _embed_component(
    score_small: np.ndarray,
    hessian_small: np.ndarray,
    indices: list[int],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed row-wise score/Hessian into a larger parameter vector."""

    n = score_small.shape[0]
    score = np.zeros((n, k), dtype=float)
    hessian = np.zeros((n, k, k), dtype=float)
    idx = np.asarray(indices, dtype=int)
    score[:, idx] = score_small
    hessian[:, idx[:, None], idx[None, :]] = hessian_small
    return score, hessian


def pnb_log_score_hessian(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    *,
    link: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P+NB2 mixture log-likelihood, score, and Hessian, row by row."""

    y = _as_1d(y)
    p = X.shape[1]
    params = np.asarray(params, dtype=float)
    tau = float(params[0])
    rho = float(expit(tau))
    beta_p = params[1 : 1 + p]
    beta_nb = params[1 + p : 1 + 2 * p]
    log_alpha = float(params[-1])
    k = 1 + 2 * p + 1

    log_pois, score_pois, hess_pois = poisson_log_score_hessian(
        y, X, beta_p, link
    )
    log_nb, score_nb, hess_nb = nb2_log_score_hessian(
        y, X, beta_nb, log_alpha, link
    )

    log_components = np.column_stack([
        np.log(rho) + log_pois,
        np.log1p(-rho) + log_nb,
    ])
    scores = np.zeros((len(y), 2, k), dtype=float)
    hessians = np.zeros((len(y), 2, k, k), dtype=float)
    tau_hess = -rho * (1.0 - rho)

    scores[:, 0, 0] = 1.0 - rho
    hessians[:, 0, 0, 0] = tau_hess
    pois_score_big, pois_hess_big = _embed_component(
        score_pois, hess_pois, list(range(1, 1 + p)), k
    )
    scores[:, 0, :] += pois_score_big
    hessians[:, 0, :, :] += pois_hess_big

    scores[:, 1, 0] = -rho
    hessians[:, 1, 0, 0] = tau_hess
    nb_indices = list(range(1 + p, 1 + 2 * p)) + [k - 1]
    nb_score_big, nb_hess_big = _embed_component(score_nb, hess_nb, nb_indices, k)
    scores[:, 1, :] += nb_score_big
    hessians[:, 1, :, :] += nb_hess_big

    return _component_mixture_log_score_hessian(log_components, scores, hessians)


def pnb_loglike_gradient_hessian(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    logp, score, hessian = pnb_log_score_hessian(params, y, X, link=link)
    return float(logp.sum()), score.sum(axis=0), hessian.sum(axis=0)


def zpnb_loglike_gradient_hessian(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Shared-inflation Z+P+NB2 Hessian.

    Parameter order is ``[tau, beta_p, beta_nb, gamma, log_alpha]``.
    """

    p, q = X.shape[1], Z.shape[1]
    params = np.asarray(params, dtype=float)
    # Count mixture in order [tau, beta_p, beta_nb, log_alpha].
    pnb_params = np.r_[params[0 : 1 + 2 * p], params[-1]]
    count_logp, count_score, count_hessian = pnb_log_score_hessian(
        pnb_params, y, X, link=link
    )
    logp, score, hessian = _zero_inflated_log_score_hessian(
        y, Z, count_logp, count_score, count_hessian, params[1 + 2 * p : 1 + 2 * p + q]
    )
    # Wrapper order is [tau, beta_p, beta_nb, log_alpha, gamma]; convert to
    # [tau, beta_p, beta_nb, gamma, log_alpha].
    k_count = 1 + 2 * p + 1
    perm = list(range(1 + 2 * p)) + list(range(k_count, k_count + q)) + [k_count - 1]
    score = score[:, perm]
    hessian = hessian[:, perm][:, :, perm]
    return float(logp.sum()), score.sum(axis=0), hessian.sum(axis=0)


def full_zip_zinb_loglike_gradient_hessian(
    params: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str = "log",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Full ZIP--ZINB observed log-likelihood, gradient, and Hessian."""

    y = _as_1d(y)
    p, q = X.shape[1], Z.shape[1]
    logit_rho, beta_z, gamma_z, beta_n, gamma_n, log_alpha = unpack_full_params(
        params, p, q
    )
    rho = float(expit(logit_rho))
    zip_logp, zip_score, zip_hessian = zip_log_score_hessian(
        y, X, Z, beta_z, gamma_z, link
    )
    zinb_logp, zinb_score, zinb_hessian = zinb_log_score_hessian(
        y, X, Z, beta_n, gamma_n, log_alpha, link
    )

    k_zip = p + q
    k_zinb = p + q + 1
    k = 1 + k_zip + k_zinb
    log_components = np.column_stack([
        np.log(rho) + zip_logp,
        np.log1p(-rho) + zinb_logp,
    ])
    scores = np.zeros((len(y), 2, k), dtype=float)
    hessians = np.zeros((len(y), 2, k, k), dtype=float)
    tau_hess = -rho * (1.0 - rho)

    scores[:, 0, 0] = 1.0 - rho
    hessians[:, 0, 0, 0] = tau_hess
    z_score_big, z_hess_big = _embed_component(
        zip_score, zip_hessian, list(range(1, 1 + k_zip)), k
    )
    scores[:, 0, :] += z_score_big
    hessians[:, 0, :, :] += z_hess_big

    scores[:, 1, 0] = -rho
    hessians[:, 1, 0, 0] = tau_hess
    n_score_big, n_hess_big = _embed_component(
        zinb_score, zinb_hessian, list(range(1 + k_zip, k)), k
    )
    scores[:, 1, :] += n_score_big
    hessians[:, 1, :, :] += n_hess_big

    logp, score, hessian = _component_mixture_log_score_hessian(
        log_components, scores, hessians
    )
    return float(logp.sum()), score.sum(axis=0), hessian.sum(axis=0)


def finite_difference_hessian_from_gradient(
    grad_fun,
    params: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central finite-difference Hessian for validation only."""

    params = np.asarray(params, dtype=float)
    k = params.size
    hessian = np.zeros((k, k), dtype=float)
    for j in range(k):
        step = np.zeros(k, dtype=float)
        step[j] = eps
        gp = grad_fun(params + step)
        gm = grad_fun(params - step)
        hessian[:, j] = (gp - gm) / (2.0 * eps)
    return 0.5 * (hessian + hessian.T)


def finite_difference_gradient(fun, params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient for validation only."""

    params = np.asarray(params, dtype=float)
    grad = np.zeros_like(params)
    for j in range(params.size):
        step = np.zeros_like(params)
        step[j] = eps
        grad[j] = (fun(params + step) - fun(params - step)) / (2.0 * eps)
    return grad
