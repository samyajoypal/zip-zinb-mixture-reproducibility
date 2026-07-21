"""Validate first-pass analytical gradients against finite differences."""

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
)


def main() -> None:
    rng = np.random.default_rng(20260710)
    n = 80
    x = rng.normal(size=n)
    X = add_intercept(x)
    Z = add_intercept(x)

    # Data do not need to be generated from the same parameters for gradient
    # validation; they only need to be valid non-negative counts.
    y = rng.poisson(np.exp(0.2 + 0.4 * x)).astype(float)

    params = np.array(
        [
            0.25,  # logit rho
            0.10,
            0.25,  # beta ZIP
            -0.40,
            0.30,  # gamma ZIP
            0.35,
            -0.10,  # beta ZINB
            -0.20,
            0.15,  # gamma ZINB
            -0.50,  # log alpha
        ],
        dtype=float,
    )

    def objective(par: np.ndarray) -> float:
        ll, _ = full_zip_zinb_loglike_gradient(par, y, X, Z, link="log")
        return ll

    ll, grad = full_zip_zinb_loglike_gradient(params, y, X, Z, link="log")
    fd = finite_difference_gradient(objective, params)

    max_abs = float(np.max(np.abs(grad - fd)))
    rel = float(max_abs / max(1.0, np.max(np.abs(fd))))

    print(f"log-likelihood: {ll:.10f}")
    print(f"max abs gradient diff: {max_abs:.3e}")
    print(f"relative gradient diff: {rel:.3e}")
    if max_abs > 5e-5 and rel > 5e-6:
        raise SystemExit("Analytical gradient validation failed.")
    print("Analytical gradient validation passed.")


if __name__ == "__main__":
    main()
