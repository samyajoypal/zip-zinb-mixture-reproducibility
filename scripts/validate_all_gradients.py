"""Validate analytical gradients for every candidate model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analytic_core import (  # noqa: E402
    add_intercept,
    finite_difference_gradient,
    full_zip_zinb_loglike_gradient,
    nb2_loglike_gradient,
    pnb_loglike_gradient,
    poisson_loglike_gradient,
    zinb_loglike_gradient,
    zip_loglike_gradient,
    zpnb_loglike_gradient,
)


def check(name: str, fun, params: np.ndarray, y, X, Z=None, link="log") -> None:
    if Z is None:
        objective = lambda par: fun(par, y, X, link=link)[0]
        ll, grad = fun(params, y, X, link=link)
    else:
        objective = lambda par: fun(par, y, X, Z, link=link)[0]
        ll, grad = fun(params, y, X, Z, link=link)
    fd = finite_difference_gradient(objective, params)
    max_abs = float(np.max(np.abs(grad - fd)))
    rel = float(max_abs / max(1.0, np.max(np.abs(fd))))
    print(f"{name:10s} ll={ll: .6f} max_abs={max_abs:.3e} rel={rel:.3e}")
    if max_abs > 1e-4 and rel > 1e-5:
        raise SystemExit(f"Gradient validation failed for {name}")


def main() -> None:
    rng = np.random.default_rng(20260710)
    n = 90
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

    check("Poisson", poisson_loglike_gradient, beta_p, y, X)
    check("NB", nb2_loglike_gradient, np.r_[beta_n, log_alpha], y, X)
    check("ZIP", zip_loglike_gradient, np.r_[beta_p, gamma_p], y, X, Z)
    check("ZINB", zinb_loglike_gradient, np.r_[beta_n, gamma_n, log_alpha], y, X, Z)
    check("P+NB", pnb_loglike_gradient, np.r_[tau, beta_p, beta_n, log_alpha], y, X)
    check("Z+P+NB", zpnb_loglike_gradient, np.r_[tau, beta_p, beta_n, gamma_p, log_alpha], y, X, Z)
    check(
        "ZIP--ZINB",
        full_zip_zinb_loglike_gradient,
        np.r_[tau, beta_p, gamma_p, beta_n, gamma_n, log_alpha],
        y,
        X,
        Z,
    )
    print("All analytical gradient validations passed.")


if __name__ == "__main__":
    main()
