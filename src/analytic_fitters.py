"""Analytical-gradient fitters for the ZIP--ZINB project.

The functions here are deliberately independent from the copied legacy model
classes.  Every model uses an analytical score vector from ``analytic_core``.
For identity-link fits, positivity is enforced with linear constraints on the
unique observed design rows rather than by clipping the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize

from analytic_core import (
    EPS,
    full_zip_zinb_loglike_gradient,
    full_zip_zinb_loglike_gradient_hessian,
    nb2_loglike_gradient,
    nb2_loglike_gradient_hessian,
    pnb_loglike_gradient,
    pnb_loglike_gradient_hessian,
    poisson_loglike_gradient,
    poisson_loglike_gradient_hessian,
    zinb_loglike_gradient,
    zinb_loglike_gradient_hessian,
    zip_loglike_gradient,
    zip_loglike_gradient_hessian,
    zpnb_loglike_gradient,
    zpnb_loglike_gradient_hessian,
)


MODEL_ORDER = (
    "Poisson",
    "NB",
    "ZIP",
    "ZINB",
    "P+NB",
    "Z+P+NB",
    "ZIP--ZINB",
)

LINKS = ("identity", "log")

GRAD_FUNCS: dict[str, Callable] = {
    "Poisson": poisson_loglike_gradient,
    "NB": nb2_loglike_gradient,
    "ZIP": zip_loglike_gradient,
    "ZINB": zinb_loglike_gradient,
    "P+NB": pnb_loglike_gradient,
    "Z+P+NB": zpnb_loglike_gradient,
    "ZIP--ZINB": full_zip_zinb_loglike_gradient,
}

HESSIAN_FUNCS: dict[str, Callable] = {
    "Poisson": poisson_loglike_gradient_hessian,
    "NB": nb2_loglike_gradient_hessian,
    "ZIP": zip_loglike_gradient_hessian,
    "ZINB": zinb_loglike_gradient_hessian,
    "P+NB": pnb_loglike_gradient_hessian,
    "Z+P+NB": zpnb_loglike_gradient_hessian,
    "ZIP--ZINB": full_zip_zinb_loglike_gradient_hessian,
}


@dataclass
class FitResult:
    model: str
    link: str
    params: np.ndarray
    llf: float
    aic: float
    bic: float
    nobs: int
    success: bool
    message: str
    nit: int
    grad_norm: float
    hessian: np.ndarray | None = None
    covariance: np.ndarray | None = None
    se: np.ndarray | None = None
    cov_status: str = "not_computed"
    min_info_eig: float = np.nan
    info_rank: int = 0


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _unique_rows(X: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(X, dtype=float), axis=0)


def _alpha_start(y: np.ndarray) -> float:
    mean = float(np.mean(y))
    var = float(np.var(y, ddof=1)) if len(y) > 1 else mean
    if mean <= 0:
        return np.log(0.1)
    alpha = max((var - mean) / max(mean * mean, EPS), 1e-3)
    return float(np.log(alpha))


def _beta_start(y: np.ndarray, X: np.ndarray, link: str, *, eps: float = 1e-8) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if link == "log":
        target = np.log(np.maximum(y, 0.0) + 0.2)
    elif link == "identity":
        target = np.maximum(y, 0.0) + 0.05
    else:
        raise ValueError(link)

    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    beta = np.asarray(beta, dtype=float)
    return _make_beta_feasible(beta, X, link, eps=eps)


def _make_beta_feasible(beta: np.ndarray, X: np.ndarray, link: str, *, eps: float = 1e-8) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).copy()
    if link != "identity":
        return beta
    X_unique = _unique_rows(X)
    eta = X_unique @ beta
    min_eta = float(np.min(eta))
    if min_eta <= eps:
        beta[0] += eps - min_eta + 0.05
    return beta


def _gamma_start(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    p_zero = float(np.mean(np.asarray(y) == 0))
    gamma = np.zeros(Z.shape[1], dtype=float)
    gamma[0] = _logit(p_zero * 0.5)
    return gamma


def _n_params(model: str, p: int, q: int) -> int:
    if model == "Poisson":
        return p
    if model == "NB":
        return p + 1
    if model == "ZIP":
        return p + q
    if model == "ZINB":
        return p + q + 1
    if model == "P+NB":
        return 1 + 2 * p + 1
    if model == "Z+P+NB":
        return 1 + 2 * p + q + 1
    if model == "ZIP--ZINB":
        return 1 + p + q + p + q + 1
    raise ValueError(model)


def _beta_slices(model: str, p: int, q: int) -> list[slice]:
    if model in {"Poisson", "NB", "ZIP", "ZINB"}:
        return [slice(0, p)]
    if model in {"P+NB", "Z+P+NB"}:
        return [slice(1, 1 + p), slice(1 + p, 1 + 2 * p)]
    if model == "ZIP--ZINB":
        return [slice(1, 1 + p), slice(1 + p + q, 1 + p + q + p)]
    raise ValueError(model)


def _base_start(model: str, y: np.ndarray, X: np.ndarray, Z: np.ndarray, link: str) -> np.ndarray:
    p, q = X.shape[1], Z.shape[1]
    beta = _beta_start(y, X, link)
    beta_nb = beta.copy()
    gamma = _gamma_start(y, Z)
    log_alpha = _alpha_start(y)

    if model == "Poisson":
        return beta
    if model == "NB":
        return np.r_[beta_nb, log_alpha]
    if model == "ZIP":
        return np.r_[beta, gamma]
    if model == "ZINB":
        return np.r_[beta_nb, gamma, log_alpha]
    if model == "P+NB":
        return np.r_[0.0, beta, beta_nb, log_alpha]
    if model == "Z+P+NB":
        return np.r_[0.0, beta, beta_nb, gamma, log_alpha]
    if model == "ZIP--ZINB":
        return np.r_[0.0, beta, gamma, beta_nb, gamma, log_alpha]
    raise ValueError(model)


def _bounds(model: str, p: int, q: int, link: str) -> list[tuple[float | None, float | None]]:
    n = _n_params(model, p, q)
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * n

    def set_slice(sl: slice, lo: float, hi: float) -> None:
        for j in range(*sl.indices(n)):
            bounds[j] = (lo, hi)

    # Keep logits finite enough to avoid overflow while still allowing
    # effectively-boundary probabilities.
    if model in {"P+NB", "Z+P+NB", "ZIP--ZINB"}:
        bounds[0] = (-20.0, 20.0)

    if model in {"ZIP", "ZINB"}:
        set_slice(slice(p, p + q), -25.0, 25.0)
    elif model == "Z+P+NB":
        set_slice(slice(1 + 2 * p, 1 + 2 * p + q), -25.0, 25.0)
    elif model == "ZIP--ZINB":
        set_slice(slice(1 + p, 1 + p + q), -25.0, 25.0)
        set_slice(slice(1 + p + q + p, 1 + p + q + p + q), -25.0, 25.0)

    if model in {"NB", "ZINB", "P+NB", "Z+P+NB", "ZIP--ZINB"}:
        bounds[-1] = (-20.0, 10.0)

    if link == "log":
        for sl in _beta_slices(model, p, q):
            set_slice(sl, -50.0, 50.0)
    return bounds


def _link_constraints(model: str, X: np.ndarray, p: int, q: int, n_params: int, link: str):
    X_unique = _unique_rows(X)
    constraints = []
    if link == "identity":
        lower = np.full(X_unique.shape[0], 1e-8)
        upper = np.full(X_unique.shape[0], np.inf)
    elif link == "log":
        # Avoid numerical overflow during optimization. This constrains
        # fitted log-means on the observed covariate grid, not coefficients
        # directly. exp(30) is already far beyond any plausible count mean here.
        lower = np.full(X_unique.shape[0], -30.0)
        upper = np.full(X_unique.shape[0], 30.0)
    else:
        raise ValueError(link)
    for sl in _beta_slices(model, p, q):
        A = np.zeros((X_unique.shape[0], n_params), dtype=float)
        A[:, sl] = X_unique
        constraints.append(optimize.LinearConstraint(A, lower, upper))
    return constraints


def _gamma_slices(model: str, p: int, q: int) -> list[slice]:
    if model in {"ZIP", "ZINB"}:
        return [slice(p, p + q)]
    if model == "Z+P+NB":
        return [slice(1 + 2 * p, 1 + 2 * p + q)]
    if model == "ZIP--ZINB":
        return [
            slice(1 + p, 1 + p + q),
            slice(1 + p + q + p, 1 + p + q + p + q),
        ]
    return []


def _all_linear_constraints(model: str, X: np.ndarray, Z: np.ndarray, p: int, q: int, n_params: int, link: str):
    constraints = _link_constraints(model, X, p, q, n_params, link)
    Z_unique = _unique_rows(Z)
    for sl in _gamma_slices(model, p, q):
        A = np.zeros((Z_unique.shape[0], n_params), dtype=float)
        A[:, sl] = Z_unique
        constraints.append(
            optimize.LinearConstraint(
                A,
                np.full(Z_unique.shape[0], -25.0),
                np.full(Z_unique.shape[0], 25.0),
            )
        )
    return constraints


def _make_start_feasible(params: np.ndarray, model: str, X: np.ndarray, Z: np.ndarray, link: str) -> np.ndarray:
    if link != "identity":
        return np.asarray(params, dtype=float)
    p, q = X.shape[1], Z.shape[1]
    params = np.asarray(params, dtype=float).copy()
    for sl in _beta_slices(model, p, q):
        params[sl] = _make_beta_feasible(params[sl], X, link)
    return params


def _jittered_starts(
    base: np.ndarray,
    model: str,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    link: str,
    *,
    n_starts: int,
    seed: int,
    extra_starts: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [np.asarray(base, dtype=float)]
    if extra_starts:
        starts.extend(np.asarray(s, dtype=float) for s in extra_starts)

    p, q = X.shape[1], Z.shape[1]
    alpha0 = _alpha_start(y)
    beta0 = _beta_start(y, X, link)
    gamma0 = _gamma_start(y, Z)

    # A few structured mixture starts help avoid the worst local optima.
    if model == "P+NB":
        for tau in (-2.0, 0.0, 2.0):
            starts.append(np.r_[tau, beta0, beta0, alpha0])
    elif model == "Z+P+NB":
        for tau in (-2.0, 0.0, 2.0):
            starts.append(np.r_[tau, beta0, beta0, gamma0, alpha0])
    elif model == "ZIP--ZINB":
        for tau in (-2.0, 0.0, 2.0):
            starts.append(np.r_[tau, beta0, gamma0, beta0, gamma0, alpha0])

    while len(starts) < n_starts:
        scale = np.full(base.size, 0.25)
        if model in {"P+NB", "Z+P+NB", "ZIP--ZINB"}:
            scale[0] = 1.0
        if model in {"NB", "ZINB", "P+NB", "Z+P+NB", "ZIP--ZINB"}:
            scale[-1] = 0.5
        starts.append(base + rng.normal(scale=scale))

    return [
        _make_start_feasible(s, model, X, Z, link)
        for s in starts[:n_starts]
    ]


def covariance_from_hessian(
    hessian: np.ndarray,
    *,
    rtol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, str, float, int]:
    """Observed-information covariance from an analytical Hessian.

    The log-likelihood Hessian should be negative semidefinite at a maximum, so
    the observed information is ``-H``. Mixture and zero-inflated models often
    land on or near boundaries; in those cases the information can be singular.
    We therefore return a Moore--Penrose-style inverse on the positive
    eigenspace and mark the status.
    """

    hessian = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    info = -hessian
    info = 0.5 * (info + info.T)
    eigvals, eigvecs = np.linalg.eigh(info)
    if not np.all(np.isfinite(eigvals)):
        k = info.shape[0]
        cov = np.full((k, k), np.nan)
        se = np.full(k, np.nan)
        return cov, se, "nonfinite_information", np.nan, 0

    scale = max(1.0, float(np.max(np.abs(eigvals))))
    tol = rtol * scale
    positive = eigvals > tol
    rank = int(np.sum(positive))
    min_eig = float(np.min(eigvals))

    k = info.shape[0]
    if rank == k:
        cov = (eigvecs / eigvals) @ eigvecs.T
        status = "inverse"
    elif rank > 0:
        V = eigvecs[:, positive]
        cov = (V / eigvals[positive]) @ V.T
        if min_eig < -tol:
            status = f"indefinite_pinv_rank_{rank}_of_{k}"
        else:
            status = f"singular_pinv_rank_{rank}_of_{k}"
    else:
        cov = np.full((k, k), np.nan)
        status = "no_positive_information"

    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    se = np.where(diag >= 0.0, np.sqrt(diag), np.nan)
    return cov, se, status, min_eig, rank


def fit_model(
    model: str,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    link: str,
    n_starts: int = 8,
    maxiter: int = 2000,
    seed: int = 20260710,
    extra_starts: list[np.ndarray] | None = None,
    compute_covariance: bool = True,
) -> FitResult:
    """Fit one model with analytical gradients and return the best start."""

    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    p, q = X.shape[1], Z.shape[1]
    base = _base_start(model, y, X, Z, link)
    starts = _jittered_starts(
        base, model, y, X, Z, link,
        n_starts=n_starts, seed=seed, extra_starts=extra_starts,
    )
    grad_func = GRAD_FUNCS[model]
    hessian_func = HESSIAN_FUNCS[model]
    n_params = _n_params(model, p, q)
    bounds = _bounds(model, p, q, link)
    constraints = _all_linear_constraints(model, X, Z, p, q, n_params, link)

    best = None

    def value_grad(par: np.ndarray) -> tuple[float, np.ndarray]:
        try:
            if model in {"Poisson", "NB", "P+NB"}:
                ll, grad = grad_func(par, y, X, link=link)
            else:
                ll, grad = grad_func(par, y, X, Z, link=link)
            if not np.isfinite(ll) or not np.all(np.isfinite(grad)):
                raise FloatingPointError("non-finite objective")
            return -ll, -grad
        except Exception:
            return 1e100, np.zeros_like(par)

    for start in starts:
        start = np.clip(
            start,
            [b[0] if b[0] is not None else -np.inf for b in bounds],
            [b[1] if b[1] is not None else np.inf for b in bounds],
        )
        method = "SLSQP"
        options = {"maxiter": maxiter, "ftol": 1e-8}

        result = optimize.minimize(
            lambda par: value_grad(par)[0],
            start,
            jac=lambda par: value_grad(par)[1],
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        nll, grad = value_grad(result.x)
        llf = -float(nll)
        grad_norm = float(np.linalg.norm(grad, ord=np.inf))
        if best is None or llf > best[0]:
            best = (llf, result, grad_norm)

    assert best is not None
    llf, result, grad_norm = best
    k = n_params
    nobs = int(y.size)
    hessian = covariance = se = None
    cov_status = "not_computed"
    min_info_eig = np.nan
    info_rank = 0
    if compute_covariance:
        try:
            if model in {"Poisson", "NB", "P+NB"}:
                _, _, hessian = hessian_func(result.x, y, X, link=link)
            else:
                _, _, hessian = hessian_func(result.x, y, X, Z, link=link)
            covariance, se, cov_status, min_info_eig, info_rank = covariance_from_hessian(hessian)
        except Exception as exc:
            cov_status = f"failed: {exc!r}"

    return FitResult(
        model=model,
        link=link,
        params=np.asarray(result.x, dtype=float),
        llf=llf,
        aic=2 * k - 2 * llf,
        bic=np.log(nobs) * k - 2 * llf,
        nobs=nobs,
        success=bool(result.success),
        message=str(result.message),
        nit=int(getattr(result, "nit", -1)),
        grad_norm=grad_norm,
        hessian=hessian,
        covariance=covariance,
        se=se,
        cov_status=cov_status,
        min_info_eig=min_info_eig,
        info_rank=info_rank,
    )


def result_row(dataset: str, fit: FitResult) -> dict[str, object]:
    return {
        "Dataset": dataset,
        "Model": fit.model,
        "Link": fit.link,
        "Log-Likelihood": fit.llf,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "nobs": fit.nobs,
        "n_params": len(fit.params),
        "success": fit.success,
        "nit": fit.nit,
        "grad_norm_inf": fit.grad_norm,
        "cov_status": fit.cov_status,
        "min_info_eig": fit.min_info_eig,
        "info_rank": fit.info_rank,
        "message": fit.message,
    }
