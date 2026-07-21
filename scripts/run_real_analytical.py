"""Run analytical-gradient model fits on the four real datasets."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analytic_core import add_intercept  # noqa: E402
from analytic_fitters import LINKS, MODEL_ORDER, fit_model, result_row  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: Path
    beta_cols: tuple[str, ...]
    gamma_cols: tuple[str, ...]


DATASETS = (
    DatasetSpec(
        "C2",
        ROOT / "data" / "datasets_csv" / "C2_long.csv",
        ("dose", "dose2"),
        ("dose",),
    ),
    DatasetSpec(
        "C3",
        ROOT / "data" / "datasets_csv" / "C3_long.csv",
        ("dose", "dose2"),
        ("dose",),
    ),
    DatasetSpec(
        "H2AX1H60",
        ROOT / "data" / "H2AX_Exposure_Wise" / "dic_1h_exposure_60.csv",
        ("dose",),
        ("dose",),
    ),
    DatasetSpec(
        "H2AX4H100",
        ROOT / "data" / "H2AX_Exposure_Wise" / "dic_4h_exposure_100.csv",
        ("dose",),
        ("dose",),
    ),
)


def load_design(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(spec.path).dropna()
    y = df["dic"].to_numpy(dtype=float)
    X = add_intercept(df.loc[:, spec.beta_cols].to_numpy(dtype=float))
    Z = add_intercept(df.loc[:, spec.gamma_cols].to_numpy(dtype=float))
    return y, X, Z


def warm_starts(model: str, p: int, q: int, done: dict[tuple[str, str], object], link: str) -> list[np.ndarray]:
    starts: list[np.ndarray] = []

    def get(name: str):
        fit = done.get((name, link))
        return None if fit is None else fit.params

    poi = get("Poisson")
    nb = get("NB")
    zip_ = get("ZIP")
    zinb = get("ZINB")
    pnb = get("P+NB")
    zpnb = get("Z+P+NB")

    if model == "P+NB" and poi is not None and nb is not None:
        starts.append(np.r_[0.0, poi[:p], nb[:p], nb[-1]])

    if model == "Z+P+NB":
        gamma = None
        if zip_ is not None:
            gamma = zip_[p : p + q]
        elif zinb is not None:
            gamma = zinb[p : p + q]
        if pnb is not None and gamma is not None:
            starts.append(np.r_[pnb[0], pnb[1 : 1 + p], pnb[1 + p : 1 + 2 * p], gamma, pnb[-1]])
        if poi is not None and nb is not None and gamma is not None:
            starts.append(np.r_[0.0, poi[:p], nb[:p], gamma, nb[-1]])

    if model == "ZIP--ZINB":
        if zip_ is not None and zinb is not None:
            starts.append(np.r_[0.0, zip_[:p], zip_[p : p + q], zinb[:p], zinb[p : p + q], zinb[-1]])
        if zpnb is not None:
            starts.append(
                np.r_[
                    zpnb[0],
                    zpnb[1 : 1 + p],
                    zpnb[1 + 2 * p : 1 + 2 * p + q],
                    zpnb[1 + p : 1 + 2 * p],
                    zpnb[1 + 2 * p : 1 + 2 * p + q],
                    zpnb[-1],
                ]
            )
        if pnb is not None:
            gamma0 = np.zeros(q)
            starts.append(np.r_[pnb[0], pnb[1 : 1 + p], gamma0, pnb[1 + p : 1 + 2 * p], gamma0, pnb[-1]])

    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / "real_analytical"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    param_rows = []
    hessian_rows = []
    covariance_rows = []

    for spec in DATASETS:
        y, X, Z = load_design(spec)
        p, q = X.shape[1], Z.shape[1]
        done: dict[tuple[str, str], object] = {}
        print(f"\nDataset {spec.key}: n={len(y)}, p={p}, q={q}")

        for model in MODEL_ORDER:
            for link in LINKS:
                print(f"  fitting {model:9s} {link:8s}", flush=True)
                extra = warm_starts(model, p, q, done, link)
                try:
                    fit = fit_model(
                        model,
                        y,
                        X,
                        Z,
                        link=link,
                        n_starts=args.n_starts,
                        maxiter=args.maxiter,
                        seed=args.seed + 17 * len(rows),
                        extra_starts=extra,
                    )
                    done[(model, link)] = fit
                    rows.append(result_row(spec.key, fit))
                    for j, value in enumerate(fit.params):
                        param_rows.append(
                            {
                                "Dataset": spec.key,
                                "Model": model,
                                "Link": link,
                                "Index": j,
                                "Value": value,
                                "SE": np.nan if fit.se is None else fit.se[j],
                                "CovarianceStatus": fit.cov_status,
                            }
                        )
                    if fit.hessian is not None:
                        for r in range(fit.hessian.shape[0]):
                            for c in range(fit.hessian.shape[1]):
                                hessian_rows.append(
                                    {
                                        "Dataset": spec.key,
                                        "Model": model,
                                        "Link": link,
                                        "Row": r,
                                        "Col": c,
                                        "Value": fit.hessian[r, c],
                                    }
                                )
                    if fit.covariance is not None:
                        for r in range(fit.covariance.shape[0]):
                            for c in range(fit.covariance.shape[1]):
                                covariance_rows.append(
                                    {
                                        "Dataset": spec.key,
                                        "Model": model,
                                        "Link": link,
                                        "Row": r,
                                        "Col": c,
                                        "Value": fit.covariance[r, c],
                                        "CovarianceStatus": fit.cov_status,
                                    }
                                )
                    print(
                        f"    ll={fit.llf:.6f} AIC={fit.aic:.3f} "
                        f"BIC={fit.bic:.3f} success={fit.success} "
                        f"grad={fit.grad_norm:.2e} cov={fit.cov_status}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "Dataset": spec.key,
                            "Model": model,
                            "Link": link,
                            "Error": repr(exc),
                        }
                    )
                    print(f"    FAILED: {exc!r}", flush=True)

    selection = pd.DataFrame(rows)
    params = pd.DataFrame(param_rows)
    hessians = pd.DataFrame(hessian_rows)
    covariances = pd.DataFrame(covariance_rows)
    failures_df = pd.DataFrame(failures)

    selection.to_csv(out_dir / "model_selection_real_analytical.csv", index=False)
    params.to_csv(out_dir / "params_real_analytical.csv", index=False)
    hessians.to_csv(out_dir / "hessian_real_analytical.csv", index=False)
    covariances.to_csv(out_dir / "covariance_real_analytical.csv", index=False)
    failures_df.to_csv(out_dir / "failures_real_analytical.csv", index=False)

    print("\nSaved:")
    print(out_dir / "model_selection_real_analytical.csv")
    print(out_dir / "params_real_analytical.csv")
    print(out_dir / "hessian_real_analytical.csv")
    print(out_dir / "covariance_real_analytical.csv")
    print(out_dir / "failures_real_analytical.csv")


if __name__ == "__main__":
    main()
