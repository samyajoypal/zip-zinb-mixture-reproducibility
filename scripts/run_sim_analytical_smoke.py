"""Small analytical simulation smoke run.

This is not the full Monte Carlo. It is a quick pipeline check that
generates ZIP--ZINB data and fits the analytical-gradient candidate models.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analytic_core import add_intercept  # noqa: E402
from analytic_fitters import MODEL_ORDER, fit_model, result_row  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    name: str
    rho: float
    alpha: float
    beta_z: np.ndarray
    beta_nb: np.ndarray
    gamma_z: np.ndarray
    gamma_nb: np.ndarray


SCENARIOS = {
    "D1": Scenario(
        name="D1_nb2_identity",
        rho=0.7,
        alpha=2.0,
        beta_z=np.array([2.0, 0.5]),
        beta_nb=np.array([4.0, 3.5]),
        gamma_z=np.array([-1.0, 0.03]),
        gamma_nb=np.array([1.0, -0.9]),
    ),
    "D2": Scenario(
        name="D2_nb2_identity",
        rho=0.5,
        alpha=1.0,
        beta_z=np.array([5.0, -0.5]),
        beta_nb=np.array([10.0, 2.5]),
        gamma_z=np.array([0.1, 3.1]),
        gamma_nb=np.array([-1.7, 2.0]),
    ),
    "D3": Scenario(
        name="D3_nb2_identity",
        rho=0.4,
        alpha=1.0,
        beta_z=np.array([6.6, -2.1]),
        beta_nb=np.array([8.0, 2.5]),
        gamma_z=np.array([-2.5, 2.0]),
        gamma_nb=np.array([-2.5, 2.5]),
    ),
}


def generate(sc: Scenario, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dose = rng.uniform(0.0, 3.0, size=n)
    X = add_intercept(dose)
    Z = X.copy()

    lam_z = X @ sc.beta_z
    lam_nb = X @ sc.beta_nb
    if np.any(lam_z <= 0) or np.any(lam_nb <= 0):
        raise ValueError("Scenario has non-positive identity-link means.")

    pi_z = expit(Z @ sc.gamma_z)
    pi_nb = expit(Z @ sc.gamma_nb)

    from_zip = rng.random(n) < sc.rho
    y = np.zeros(n, dtype=float)

    zip_count = from_zip & (rng.random(n) >= pi_z)
    y[zip_count] = rng.poisson(lam_z[zip_count])

    nb_count = (~from_zip) & (rng.random(n) >= pi_nb)
    r = 1.0 / sc.alpha
    p = r / (r + lam_nb[nb_count])
    y[nb_count] = rng.negative_binomial(r, p)
    return y, X, Z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="D1")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    sc = SCENARIOS[args.scenario]
    out_dir = ROOT / "outputs" / "sim_analytical"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    param_rows = []
    for rep in range(args.reps):
        y, X, Z = generate(sc, args.n, args.seed + rep)
        print(f"rep={rep} scenario={sc.name} n={args.n}")
        for model in MODEL_ORDER:
            fit = fit_model(
                model,
                y,
                X,
                Z,
                link="identity",
                n_starts=args.n_starts,
                maxiter=args.maxiter,
                seed=args.seed + 1000 * rep + len(rows),
            )
            row = result_row(sc.name, fit)
            row["Rep"] = rep
            rows.append(row)
            for j, value in enumerate(fit.params):
                param_rows.append(
                    {
                        "Scenario": sc.name,
                        "Rep": rep,
                        "Model": model,
                        "Index": j,
                        "Value": value,
                    }
                )
            print(f"  {model:9s} ll={fit.llf:.3f} AIC={fit.aic:.3f} success={fit.success}")

    pd.DataFrame(rows).to_csv(out_dir / "smoke_model_selection.csv", index=False)
    pd.DataFrame(param_rows).to_csv(out_dir / "smoke_params.csv", index=False)
    print(out_dir / "smoke_model_selection.csv")
    print(out_dir / "smoke_params.csv")


if __name__ == "__main__":
    main()
