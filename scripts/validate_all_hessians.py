"""Validate analytical Hessians against finite differences of gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analytic_core import (  # noqa: E402
    add_intercept,
    finite_difference_hessian_from_gradient,
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


def check(name: str, grad_fun, hess_fun, params: np.ndarray, y, X, Z=None, link="log") -> None:
    if Z is None:
        grad_only = lambda par: grad_fun(par, y, X, link=link)[1]
        ll, grad, hess = hess_fun(params, y, X, link=link)
    else:
        grad_only = lambda par: grad_fun(par, y, X, Z, link=link)[1]
        ll, grad, hess = hess_fun(params, y, X, Z, link=link)
    fd = finite_difference_hessian_from_gradient(grad_only, params)
    max_abs = float(np.max(np.abs(hess - fd)))
    rel = float(max_abs / max(1.0, np.max(np.abs(fd))))
    sym = float(np.max(np.abs(hess - hess.T)))
    gnorm = float(np.linalg.norm(grad, ord=np.inf))
    print(
        f"{name:10s} ll={ll: .6f} grad_inf={gnorm:.3e} "
        f"max_abs={max_abs:.3e} rel={rel:.3e} sym={sym:.3e}"
    )
    if max_abs > 5e-3 and rel > 5e-5:
        raise SystemExit(f"Hessian validation failed for {name}")


def main() -> None:
    rng = np.random.default_rng(20260710)
    n = 80
    x = rng.normal(size=n)
    X = add_intercept(x)
    Z = add_intercept(0.5 * x)
    y = rng.poisson(np.exp(0.15 + 0.35 * x)).astype(float)

    beta_p = np.array([0.2, 0.3])
    beta_n = np.array([0.45, -0.15])
    gamma_p = np.array([-0.5, 0.25])
    gamma_n = np.array([-0.35, -0.1])
    tau = np.array([0.2])
    log_alpha = np.array([-0.4])

    check("Poisson", poisson_loglike_gradient, poisson_loglike_gradient_hessian, beta_p, y, X)
    check("NB", nb2_loglike_gradient, nb2_loglike_gradient_hessian, np.r_[beta_n, log_alpha], y, X)
    check("ZIP", zip_loglike_gradient, zip_loglike_gradient_hessian, np.r_[beta_p, gamma_p], y, X, Z)
    check("ZINB", zinb_loglike_gradient, zinb_loglike_gradient_hessian, np.r_[beta_n, gamma_n, log_alpha], y, X, Z)
    check("P+NB", pnb_loglike_gradient, pnb_loglike_gradient_hessian, np.r_[tau, beta_p, beta_n, log_alpha], y, X)
    check("Z+P+NB", zpnb_loglike_gradient, zpnb_loglike_gradient_hessian, np.r_[tau, beta_p, beta_n, gamma_p, log_alpha], y, X, Z)
    check(
        "ZIP--ZINB",
        full_zip_zinb_loglike_gradient,
        full_zip_zinb_loglike_gradient_hessian,
        np.r_[tau, beta_p, gamma_p, beta_n, gamma_n, log_alpha],
        y,
        X,
        Z,
    )
    print("All analytical Hessian validations passed.")


if __name__ == "__main__":
    main()
