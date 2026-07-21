"""Full analytical ZIP--ZINB simulation pipeline.

Run from the independent project root, for example:

    /home/samyajoypal/.local/share/pipx/venvs/spyder/bin/python \
        scripts/run_simulation_analytical_full.py

The script is designed for long local runs:

* parallel execution across scenario/replicate tasks;
* a visible tqdm progress bar with ETA;
* checkpoint/resume via ``completed_reps.csv``;
* LaTeX tables matching the active simulation summaries:
    - ``tab:sim``
    - ``tab:sim_more``
    - ``tab:sim_selection``
    - ``tab:sim_lrt``

The data-generating mechanism uses identity-link means that remain positive on
the simulation design range.  The analytical fitted models enforce positivity
of fitted identity-link means on the observed design grid.
"""

from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/zip_zinb_matplotlib")

import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.stats import chi2
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analytic_core import add_intercept  # noqa: E402
from analytic_fitters import GRAD_FUNCS, MODEL_ORDER, fit_model, result_row  # noqa: E402


LINK = "identity"
EPS_MEAN = 1e-12
FULL_MODEL = "ZIP--ZINB"
LRT_MODELS = ("Poisson", "NB", "P+NB", "ZIP", "ZINB", "Z+P+NB")
Y_PARAM_ORDER = ("rho", "lambda", "pi", "varphi", "alpha", "omega")
COEF_ORDER = (
    ("beta_z[0]", r"$\beta_{10}$"),
    ("beta_z[1]", r"$\beta_{11}$"),
    ("gamma_z[0]", r"$\gamma_{10}$"),
    ("gamma_z[1]", r"$\gamma_{11}$"),
    ("beta_nb[0]", r"$\beta_{20}$"),
    ("beta_nb[1]", r"$\beta_{21}$"),
    ("gamma_nb[0]", r"$\gamma_{20}$"),
    ("gamma_nb[1]", r"$\gamma_{21}$"),
)

LRT_BOUNDARY_DIMS = {
    "Poisson": 2,
    "NB": 2,
    "P+NB": 2,
    "ZIP": 1,
    "ZINB": 1,
    "Z+P+NB": 0,
}


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    n: int
    dose_min: float
    dose_max: float
    rho: float
    alpha: float
    beta_z: tuple[float, float]
    beta_nb: tuple[float, float]
    gamma_z: tuple[float, float]
    gamma_nb: tuple[float, float]
    base_seed: int

    @property
    def beta_z_arr(self) -> np.ndarray:
        return np.asarray(self.beta_z, dtype=float)

    @property
    def beta_nb_arr(self) -> np.ndarray:
        return np.asarray(self.beta_nb, dtype=float)

    @property
    def gamma_z_arr(self) -> np.ndarray:
        return np.asarray(self.gamma_z, dtype=float)

    @property
    def gamma_nb_arr(self) -> np.ndarray:
        return np.asarray(self.gamma_nb, dtype=float)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="D1",
        label="Dataset 1",
        n=20_000,
        dose_min=0.0,
        dose_max=3.0,
        rho=0.7,
        alpha=1.5,
        beta_z=(2.0, 1.0),
        beta_nb=(4.0, 3.5),
        gamma_z=(-2.0, 1.0),
        gamma_nb=(1.0, -3.0),
        base_seed=501,
    ),
    Scenario(
        key="D2",
        label="Dataset 2",
        n=20_000,
        dose_min=0.0,
        dose_max=3.0,
        rho=0.5,
        alpha=1.0,
        beta_z=(5.0, -0.5),
        beta_nb=(10.0, 2.5),
        gamma_z=(0.1, 3.1),
        gamma_nb=(-1.7, 2.0),
        base_seed=10,
    ),
    Scenario(
        key="D3",
        label="Dataset 3",
        n=20_000,
        dose_min=0.0,
        dose_max=3.0,
        rho=0.4,
        alpha=1.0,
        # Revised from (7.0, -2.5): the old specification made lambda_i<0
        # for x_i>2.8 under the identity link.  The new coefficients preserve
        # the decreasing ZIP mean while keeping lambda_i>=0.3 on [0,3].
        beta_z=(6.6, -2.1),
        beta_nb=(8.0, 2.5),
        gamma_z=(-2.5, 2.0),
        gamma_nb=(-2.5, 2.5),
        base_seed=156,
    ),
    Scenario(
        key="D4",
        label="Dataset 4",
        n=20_000,
        dose_min=0.0,
        dose_max=3.0,
        rho=0.3,
        alpha=1.0,
        beta_z=(5.0, 1.5),
        beta_nb=(6.0, -1.0),
        gamma_z=(-2.0, 1.0),
        gamma_nb=(-3.70, 2.0),
        base_seed=156,
    ),
)


SCENARIO_BY_KEY = {sc.key: sc for sc in SCENARIOS}
SCENARIO_BY_LABEL = {sc.label: sc for sc in SCENARIOS}


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return float(np.log(p / (1.0 - p)))


def rep_seed(sc: Scenario, rep: int, global_seed: int) -> int:
    return int(global_seed + sc.base_seed * 1_000_003 + rep * 9_176)


def inv_identity_clipped(eta: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(eta, dtype=float), EPS_MEAN, None)


def generate_dataset(sc: Scenario, n: int, seed: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    dose = rng.uniform(sc.dose_min, sc.dose_max, size=n)
    X = add_intercept(dose)
    Z = X.copy()

    y, lam, pi, varphi, omega = simulate_response_given_design(sc, X, Z, rng)

    return {
        "dose": dose,
        "X": X,
        "Z": Z,
        "y": y,
        "lambda": lam,
        "pi": pi,
        "varphi": varphi,
        "omega": omega,
        "rho": float(sc.rho),
        "alpha": float(sc.alpha),
    }


def simulate_response_given_design(
    sc: Scenario,
    X: np.ndarray,
    Z: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lam = inv_identity_clipped(X @ sc.beta_z_arr)
    varphi = inv_identity_clipped(X @ sc.beta_nb_arr)
    pi = expit(Z @ sc.gamma_z_arr)
    omega = expit(Z @ sc.gamma_nb_arr)
    n = X.shape[0]

    from_zip = rng.random(n) < sc.rho
    y_zip = rng.poisson(lam)
    y_zip[rng.random(n) < pi] = 0

    r_nb = 1.0 / sc.alpha
    p_nb = r_nb / (r_nb + varphi)
    y_nb = rng.negative_binomial(r_nb, p_nb)
    y_nb[rng.random(n) < omega] = 0

    y = np.empty(n, dtype=float)
    y[from_zip] = y_zip[from_zip]
    y[~from_zip] = y_nb[~from_zip]

    return y, lam, pi, varphi, omega


def true_yparams(sc: Scenario, data: dict[str, np.ndarray | float]) -> dict[str, float]:
    return {
        "rho": float(sc.rho),
        "lambda": float(np.mean(data["lambda"])),
        "pi": float(np.mean(data["pi"])),
        "varphi": float(np.mean(data["varphi"])),
        "alpha": float(sc.alpha),
        "omega": float(np.mean(data["omega"])),
    }


def unpack_full(theta: np.ndarray) -> dict[str, np.ndarray | float]:
    theta = np.asarray(theta, dtype=float)
    return {
        "logit_rho": float(theta[0]),
        "rho": float(expit(theta[0])),
        "beta_z": theta[1:3],
        "gamma_z": theta[3:5],
        "beta_nb": theta[5:7],
        "gamma_nb": theta[7:9],
        "log_alpha": float(theta[9]),
        "alpha": float(np.exp(theta[9])),
    }


def estimated_yparams(theta: np.ndarray, X: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    par = unpack_full(theta)
    lam = inv_identity_clipped(X @ par["beta_z"])
    varphi = inv_identity_clipped(X @ par["beta_nb"])
    pi = expit(Z @ par["gamma_z"])
    omega = expit(Z @ par["gamma_nb"])
    return {
        "rho": float(par["rho"]),
        "lambda": float(np.mean(lam)),
        "pi": float(np.mean(pi)),
        "varphi": float(np.mean(varphi)),
        "alpha": float(par["alpha"]),
        "omega": float(np.mean(omega)),
    }


def true_coefs(sc: Scenario) -> dict[str, float]:
    return {
        "beta_z[0]": sc.beta_z[0],
        "beta_z[1]": sc.beta_z[1],
        "gamma_z[0]": sc.gamma_z[0],
        "gamma_z[1]": sc.gamma_z[1],
        "beta_nb[0]": sc.beta_nb[0],
        "beta_nb[1]": sc.beta_nb[1],
        "gamma_nb[0]": sc.gamma_nb[0],
        "gamma_nb[1]": sc.gamma_nb[1],
    }


def estimated_coefs(theta: np.ndarray) -> dict[str, float]:
    par = unpack_full(theta)
    return {
        "beta_z[0]": float(par["beta_z"][0]),
        "beta_z[1]": float(par["beta_z"][1]),
        "gamma_z[0]": float(par["gamma_z"][0]),
        "gamma_z[1]": float(par["gamma_z"][1]),
        "beta_nb[0]": float(par["beta_nb"][0]),
        "beta_nb[1]": float(par["beta_nb"][1]),
        "gamma_nb[0]": float(par["gamma_nb"][0]),
        "gamma_nb[1]": float(par["gamma_nb"][1]),
    }


def true_full_params(sc: Scenario) -> tuple[list[str], np.ndarray]:
    names = [
        "logit_rho",
        "beta_z[0]",
        "beta_z[1]",
        "gamma_z[0]",
        "gamma_z[1]",
        "beta_nb[0]",
        "beta_nb[1]",
        "gamma_nb[0]",
        "gamma_nb[1]",
        "log_alpha",
    ]
    values = np.r_[
        logit(sc.rho),
        sc.beta_z_arr,
        sc.gamma_z_arr,
        sc.beta_nb_arr,
        sc.gamma_nb_arr,
        np.log(sc.alpha),
    ]
    return names, values.astype(float)


def loglik_for_fit(model: str, params: np.ndarray, y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> float:
    fun = GRAD_FUNCS[model]
    if model in {"Poisson", "NB", "P+NB"}:
        ll, _ = fun(params, y, X, link=LINK)
    else:
        ll, _ = fun(params, y, X, Z, link=LINK)
    return float(ll)


def truth_start(model: str, sc: Scenario) -> np.ndarray:
    beta_z = sc.beta_z_arr
    beta_nb = sc.beta_nb_arr
    gamma_z = sc.gamma_z_arr
    gamma_nb = sc.gamma_nb_arr
    log_alpha = float(np.log(sc.alpha))
    tau = logit(sc.rho)
    if model == "Poisson":
        return beta_z
    if model == "NB":
        return np.r_[beta_nb, log_alpha]
    if model == "ZIP":
        return np.r_[beta_z, gamma_z]
    if model == "ZINB":
        return np.r_[beta_nb, gamma_nb, log_alpha]
    if model == "P+NB":
        return np.r_[tau, beta_z, beta_nb, log_alpha]
    if model == "Z+P+NB":
        return np.r_[tau, beta_z, beta_nb, 0.5 * (gamma_z + gamma_nb), log_alpha]
    if model == "ZIP--ZINB":
        return np.r_[tau, beta_z, gamma_z, beta_nb, gamma_nb, log_alpha]
    raise ValueError(model)


def warm_starts(model: str, done: dict[str, object], sc: Scenario, use_true_starts: bool) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    if use_true_starts:
        starts.append(truth_start(model, sc))

    def get(name: str):
        fit = done.get(name)
        return None if fit is None else fit.params

    poi = get("Poisson")
    nb = get("NB")
    zip_ = get("ZIP")
    zinb = get("ZINB")
    pnb = get("P+NB")
    zpnb = get("Z+P+NB")

    p = q = 2
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


def chibar_pvalue(stat: float, k: int) -> float:
    dfs = np.arange(k + 1, dtype=int)
    weights = np.array([math.comb(k, int(df)) for df in dfs], dtype=float) / (2.0**k)
    tails = chi2.sf(max(0.0, stat), dfs).astype(float)
    tails[dfs == 0] = 1.0 if stat <= 0.0 else 0.0
    return float(np.dot(weights, tails))


def lrt_rows_for_fits(sc: Scenario, rep: int, fits: dict[str, object], alpha_level: float) -> list[dict[str, object]]:
    if FULL_MODEL not in fits:
        return []
    full = fits[FULL_MODEL]
    rows: list[dict[str, object]] = []
    for model in LRT_MODELS:
        if model not in fits:
            continue
        restricted = fits[model]
        stat = max(0.0, 2.0 * (full.llf - restricted.llf))
        boundary_dim = LRT_BOUNDARY_DIMS[model]
        df_diff = max(1, len(full.params) - len(restricted.params))
        if boundary_dim:
            p_value = chibar_pvalue(stat, boundary_dim)
            method = f"chi-bar² heuristic; k={boundary_dim}"
            df_value = np.nan
        else:
            p_value = float(chi2.sf(stat, df_diff))
            if model == "Z+P+NB":
                method = f"chi²(df={df_diff}; shared inflation gamma_z=gamma_nb)"
            else:
                method = f"chi²(df={df_diff})"
            df_value = df_diff
        rows.append(
            {
                "Scenario": sc.key,
                "Dataset": sc.label,
                "Rep": rep,
                "RestrictedModel": model,
                "FullModel": FULL_MODEL,
                "RestrictedLogLik": restricted.llf,
                "FullLogLik": full.llf,
                "LRT": stat,
                "df": df_value,
                "boundary_dim": boundary_dim,
                "p_value": p_value,
                "reject": bool(p_value < alpha_level),
                "method": method,
            }
        )
    return rows


def run_one_task(task: dict[str, object]) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sc = SCENARIO_BY_KEY[str(task["scenario"])]
    rep = int(task["rep"])
    n = int(task["n"])
    seed = int(task["seed"])
    n_starts = int(task["n_starts"])
    maxiter = int(task["maxiter"])
    alpha_level = float(task["alpha_level"])
    use_true_starts = bool(task["use_true_starts"])
    compute_coverage = bool(task.get("compute_coverage", False))
    n_test = int(task.get("n_test", 0))

    t0 = time.time()
    data = generate_dataset(sc, n, seed)
    y = np.asarray(data["y"], dtype=float)
    X = np.asarray(data["X"], dtype=float)
    Z = np.asarray(data["Z"], dtype=float)

    fits: dict[str, object] = {}
    selection_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    for i, model in enumerate(MODEL_ORDER):
        try:
            extra_starts = warm_starts(model, fits, sc, use_true_starts)
            fit_t0 = time.time()
            fit = fit_model(
                model,
                y,
                X,
                Z,
                link=LINK,
                n_starts=n_starts,
                maxiter=maxiter,
                seed=seed + 1009 * (i + 1),
                extra_starts=extra_starts,
                compute_covariance=compute_coverage and model == FULL_MODEL,
            )
            fit_elapsed = time.time() - fit_t0
            fits[model] = fit
            row = result_row(sc.key, fit)
            row.update({"Scenario": sc.key, "DatasetLabel": sc.label, "Rep": rep, "fit_elapsed_sec": fit_elapsed})
            selection_rows.append(row)
            runtime_rows.append(
                {
                    "Scenario": sc.key,
                    "Dataset": sc.label,
                    "Rep": rep,
                    "Model": model,
                    "fit_elapsed_sec": fit_elapsed,
                    "success": bool(fit.success),
                    "nit": int(fit.nit),
                    "grad_norm_inf": float(fit.grad_norm),
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "Scenario": sc.key,
                    "Dataset": sc.label,
                    "Rep": rep,
                    "Model": model,
                    "Error": repr(exc),
                }
            )

    yparam_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    if FULL_MODEL in fits:
        full = fits[FULL_MODEL]
        y_true = true_yparams(sc, data)
        y_est = estimated_yparams(full.params, X, Z)
        for name in Y_PARAM_ORDER:
            yparam_rows.append(
                {
                    "Scenario": sc.key,
                    "Dataset": sc.label,
                    "Rep": rep,
                    "Parameter": name,
                    "true": y_true[name],
                    "estimate": y_est[name],
                }
            )

        c_true = true_coefs(sc)
        c_est = estimated_coefs(full.params)
        for name, latex in COEF_ORDER:
            coef_rows.append(
                {
                    "Scenario": sc.key,
                    "Dataset": sc.label,
                    "Rep": rep,
                    "Coefficient": name,
                    "CoefficientLatex": latex,
                    "true": c_true[name],
                    "estimate": c_est[name],
                }
            )
        for j, value in enumerate(full.params):
            param_rows.append(
                {
                    "Scenario": sc.key,
                    "Dataset": sc.label,
                    "Rep": rep,
                    "Model": FULL_MODEL,
                    "Index": j,
                    "Value": value,
                }
            )
        if compute_coverage and full.se is not None:
            names, true_values = true_full_params(sc)
            se = np.asarray(full.se, dtype=float)
            est = np.asarray(full.params, dtype=float)
            for j, name in enumerate(names):
                lo = est[j] - 1.96 * se[j]
                hi = est[j] + 1.96 * se[j]
                coverage_rows.append(
                    {
                        "Scenario": sc.key,
                        "Dataset": sc.label,
                        "Rep": rep,
                        "Parameter": name,
                        "Scale": "optimizer",
                        "true": true_values[j],
                        "estimate": est[j],
                        "se": se[j],
                        "lower95": lo,
                        "upper95": hi,
                        "covered": bool(lo <= true_values[j] <= hi),
                        "CovarianceStatus": full.cov_status,
                    }
                )
            par = unpack_full(full.params)
            rho = float(par["rho"])
            alpha_hat = float(par["alpha"])
            scalar_specs = [
                ("rho", sc.rho, rho, se[0] * rho * (1.0 - rho)),
                ("alpha", sc.alpha, alpha_hat, se[-1] * alpha_hat),
            ]
            for name, true_value, estimate, scalar_se in scalar_specs:
                lo = estimate - 1.96 * scalar_se
                hi = estimate + 1.96 * scalar_se
                coverage_rows.append(
                    {
                        "Scenario": sc.key,
                        "Dataset": sc.label,
                        "Rep": rep,
                        "Parameter": name,
                        "Scale": "natural_delta",
                        "true": true_value,
                        "estimate": estimate,
                        "se": scalar_se,
                        "lower95": lo,
                        "upper95": hi,
                        "covered": bool(lo <= true_value <= hi),
                        "CovarianceStatus": full.cov_status,
                    }
                )

    logscore_rows: list[dict[str, object]] = []
    if n_test > 0 and fits:
        test_rng = np.random.default_rng(seed + 8_000_003)
        idx = test_rng.choice(np.arange(len(y)), size=n_test, replace=True)
        X_test = X[idx, :]
        Z_test = Z[idx, :]
        y_test, *_ = simulate_response_given_design(sc, X_test, Z_test, test_rng)
        for model, fit in fits.items():
            try:
                ll_test = loglik_for_fit(model, fit.params, y_test, X_test, Z_test)
                logscore_rows.append(
                    {
                        "Scenario": sc.key,
                        "Dataset": sc.label,
                        "Rep": rep,
                        "Model": model,
                        "n_test": n_test,
                        "test_loglik": ll_test,
                        "mean_test_log_score": ll_test / max(1, n_test),
                    }
                )
            except Exception as exc:
                failure_rows.append(
                    {
                        "Scenario": sc.key,
                        "Dataset": sc.label,
                        "Rep": rep,
                        "Model": f"{model}__test_logscore",
                        "Error": repr(exc),
                    }
                )

    return {
        "ok": True,
        "Scenario": sc.key,
        "Dataset": sc.label,
        "Rep": rep,
        "elapsed_sec": time.time() - t0,
        "selection_rows": selection_rows,
        "yparam_rows": yparam_rows,
        "coef_rows": coef_rows,
        "param_rows": param_rows,
        "coverage_rows": coverage_rows,
        "logscore_rows": logscore_rows,
        "runtime_rows": runtime_rows,
        "lrt_rows": lrt_rows_for_fits(sc, rep, fits, alpha_level),
        "failure_rows": failure_rows,
    }


def append_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return {(str(row.Scenario), int(row.Rep)) for row in df.itertuples(index=False)}


def existing_or_empty(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def scenario_order_key(value: str) -> int:
    order = {sc.key: i for i, sc in enumerate(SCENARIOS)}
    return order.get(str(value), 999)


def summarise_yparams(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Scenario", "Dataset", "Parameter", "true", "mean_estimate", "empirical_sd", "rmse", "n_success"])
    rows = []
    for (scenario, dataset, param), group in df.groupby(["Scenario", "Dataset", "Parameter"], sort=False):
        est = group["estimate"].astype(float)
        true = group["true"].astype(float)
        rows.append(
            {
                "Scenario": scenario,
                "Dataset": dataset,
                "Parameter": param,
                "true": float(true.mean()),
                "mean_estimate": float(est.mean()),
                "empirical_sd": float(est.std(ddof=1)) if len(est) > 1 else 0.0,
                "rmse": float(np.sqrt(np.mean((est - true) ** 2))),
                "n_success": int(len(est)),
            }
        )
    out = pd.DataFrame(rows)
    out["ScenarioOrder"] = out["Scenario"].map(scenario_order_key)
    out["ParameterOrder"] = out["Parameter"].map({p: i for i, p in enumerate(Y_PARAM_ORDER)})
    return out.sort_values(["ScenarioOrder", "ParameterOrder"]).drop(columns=["ScenarioOrder", "ParameterOrder"]).reset_index(drop=True)


def summarise_coefs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Scenario",
                "Dataset",
                "Coefficient",
                "CoefficientLatex",
                "true",
                "mean_estimate",
                "empirical_sd",
                "rmse",
                "n_success",
            ]
        )
    rows = []
    for (scenario, dataset, coef, latex), group in df.groupby(
        ["Scenario", "Dataset", "Coefficient", "CoefficientLatex"], sort=False
    ):
        est = group["estimate"].astype(float)
        true = group["true"].astype(float)
        rows.append(
            {
                "Scenario": scenario,
                "Dataset": dataset,
                "Coefficient": coef,
                "CoefficientLatex": latex,
                "true": float(true.mean()),
                "mean_estimate": float(est.mean()),
                "empirical_sd": float(est.std(ddof=1)) if len(est) > 1 else 0.0,
                "rmse": float(np.sqrt(np.mean((est - true) ** 2))),
                "n_success": int(len(est)),
            }
        )
    out = pd.DataFrame(rows)
    out["ScenarioOrder"] = out["Scenario"].map(scenario_order_key)
    out["CoefOrder"] = out["Coefficient"].map({name: i for i, (name, _) in enumerate(COEF_ORDER)})
    return out.sort_values(["ScenarioOrder", "CoefOrder"]).drop(columns=["ScenarioOrder", "CoefOrder"]).reset_index(drop=True)


def summarise_selection(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Scenario", "Dataset", "aic_selection_rate", "bic_selection_rate", "n_reps"])
    rows = []
    for scenario, group in df.groupby("Scenario", sort=False):
        dataset = SCENARIO_BY_KEY[str(scenario)].label
        best_aic = group.loc[group.groupby("Rep")["AIC"].idxmin()]
        best_bic = group.loc[group.groupby("Rep")["BIC"].idxmin()]
        rows.append(
            {
                "Scenario": scenario,
                "Dataset": dataset,
                "aic_selection_rate": float((best_aic["Model"] == FULL_MODEL).mean()),
                "bic_selection_rate": float((best_bic["Model"] == FULL_MODEL).mean()),
                "n_reps": int(group["Rep"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out["ScenarioOrder"] = out["Scenario"].map(scenario_order_key)
    return out.sort_values("ScenarioOrder").drop(columns=["ScenarioOrder"]).reset_index(drop=True)


def summarise_lrt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Scenario", "Dataset", "RestrictedModel", "rejection_rate", "n_tests"])
    rows = []
    for (scenario, dataset, model), group in df.groupby(["Scenario", "Dataset", "RestrictedModel"], sort=False):
        rows.append(
            {
                "Scenario": scenario,
                "Dataset": dataset,
                "RestrictedModel": model,
                "rejection_rate": float(group["reject"].astype(bool).mean()),
                "n_tests": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    out["ScenarioOrder"] = out["Scenario"].map(scenario_order_key)
    out["ModelOrder"] = out["RestrictedModel"].map({m: i for i, m in enumerate(LRT_MODELS)})
    return out.sort_values(["ScenarioOrder", "ModelOrder"]).drop(columns=["ScenarioOrder", "ModelOrder"]).reset_index(drop=True)


def summarise_runtime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Scenario", "Dataset", "Model", "mean_sec", "median_sec", "sd_sec", "n_fits"])
    rows = []
    for (scenario, dataset, model), group in df.groupby(["Scenario", "Dataset", "Model"], sort=False):
        elapsed = group["fit_elapsed_sec"].astype(float)
        rows.append(
            {
                "Scenario": scenario,
                "Dataset": dataset,
                "Model": model,
                "mean_sec": float(elapsed.mean()),
                "median_sec": float(elapsed.median()),
                "sd_sec": float(elapsed.std(ddof=1)) if len(elapsed) > 1 else 0.0,
                "n_fits": int(len(elapsed)),
            }
        )
    out = pd.DataFrame(rows)
    out["ScenarioOrder"] = out["Scenario"].map(scenario_order_key)
    out["ModelOrder"] = out["Model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["ScenarioOrder", "ModelOrder"]).drop(columns=["ScenarioOrder", "ModelOrder"]).reset_index(drop=True)


def summarise_logscore(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Scenario", "Dataset", "Model", "mean_test_log_score", "sd_test_log_score", "n_reps"])
    rows = []
    for (scenario, dataset, model), group in df.groupby(["Scenario", "Dataset", "Model"], sort=False):
        scores = group["mean_test_log_score"].astype(float)
        rows.append(
            {
                "Scenario": scenario,
                "Dataset": dataset,
                "Model": model,
                "mean_test_log_score": float(scores.mean()),
                "sd_test_log_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "n_reps": int(len(scores)),
            }
        )
    out = pd.DataFrame(rows)
    out["ScenarioOrder"] = out["Scenario"].map(scenario_order_key)
    out["ModelOrder"] = out["Model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["ScenarioOrder", "ModelOrder"]).drop(columns=["ScenarioOrder", "ModelOrder"]).reset_index(drop=True)


def summarise_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Scenario", "Dataset", "Parameter", "Scale", "coverage_rate", "mean_se", "n_intervals"])
    rows = []
    for (scenario, dataset, param, scale), group in df.groupby(["Scenario", "Dataset", "Parameter", "Scale"], sort=False):
        rows.append(
            {
                "Scenario": scenario,
                "Dataset": dataset,
                "Parameter": param,
                "Scale": scale,
                "coverage_rate": float(group["covered"].astype(bool).mean()),
                "mean_se": float(group["se"].astype(float).mean()),
                "n_intervals": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    out["ScenarioOrder"] = out["Scenario"].map(scenario_order_key)
    return out.sort_values(["ScenarioOrder", "Scale", "Parameter"]).drop(columns=["ScenarioOrder"]).reset_index(drop=True)


def fmt(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.4f}"


def fmt_rate(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.2f}"


def latex_param_name(param: str) -> str:
    return {
        "rho": r"$\rho$",
        "lambda": r"$\lambda$",
        "pi": r"$\pi$",
        "varphi": r"$\varphi$",
        "alpha": r"$\alpha$",
        "omega": r"$\omega$",
    }[param]


def write_yparam_table(summary: pd.DataFrame, path: Path, n_caption: int, reps_caption: int) -> None:
    lines = [
        r"\begin{table}[!h]",
        r"\caption{Simulation results for datasets with one regressor, an intercept, and identity link function}",
        r"\begin{threeparttable}",
        r"\adjustbox{max width=\textwidth}{%",
        r"    \centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" Dataset & Parameter & True value & Mean estimate & Empirical SD \\",
        r"\midrule",
    ]
    for sidx, sc in enumerate(SCENARIOS):
        part = summary[summary.Scenario == sc.key].set_index("Parameter")
        for pidx, param in enumerate(Y_PARAM_ORDER):
            prefix = rf"\multirow{{6}}{{*}}{{{sc.label}}}" if pidx == 0 else ""
            row = part.loc[param] if param in part.index else None
            if row is None:
                values = "-- & -- & --"
            else:
                values = f"{fmt(row['true'])} & {fmt(row['mean_estimate'])} & {fmt(row['empirical_sd'])}"
            lines.append(f"{prefix} & {latex_param_name(param):<9} & {values}\\\\")
        if sidx != len(SCENARIOS) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"      \begin{tablenotes}\footnotesize",
        rf"    \item For each parameter, the table reports the Monte Carlo mean and empirical standard deviation of the estimates across {reps_caption} simulation replicates ($n={n_caption:,}$). For parameters that depend on covariates (e.g., $\lambda_i$, $\pi_i$, $\varphi_i$, $\omega_i$), the reported ``true value'' corresponds to the Monte Carlo average of the data-generating quantities implied by the fixed regression coefficients.",
        r"  \end{tablenotes}",
        r"    \end{threeparttable}",
        r"\label{tab:sim}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_coef_table(summary: pd.DataFrame, path: Path, reps_caption: int) -> None:
    lines = [
        r"\begin{table}[!h]",
        rf"\caption{{True and mean of estimated regression coefficients along with their empirical standard deviations (E.SD) and root mean squared errors (RMSE) on simulation datasets from {reps_caption} simulation replicates}}",
        r"\adjustbox{max width=\textwidth}{%",
        r"    \centering",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" Dataset & Coefficients & True value & Mean estimated value & E.SD & RMSE \\",
        r"\midrule",
    ]
    for sidx, sc in enumerate(SCENARIOS):
        part = summary[summary.Scenario == sc.key].set_index("Coefficient")
        for cidx, (coef, latex) in enumerate(COEF_ORDER):
            prefix = rf"\multirow{{8}}{{*}}{{{sc.label}}}" if cidx == 0 else ""
            row = part.loc[coef] if coef in part.index else None
            if row is None:
                values = "-- & -- & -- & --"
            else:
                values = (
                    f"{fmt(row['true'])} & {fmt(row['mean_estimate'])} & "
                    f"{fmt(row['empirical_sd'])} & {fmt(row['rmse'])}"
                )
            lines.append(f"{prefix} & {latex} & {values}\\\\")
        if sidx != len(SCENARIOS) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"    \label{tab:sim_more}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_selection_table(summary: pd.DataFrame, path: Path, reps_caption: int) -> None:
    lines = [
        r"\begin{table}[!h]",
        rf"\caption{{Model selection rate of the proposed ZIP--ZINB mixture model based on AIC and BIC across {reps_caption} simulation replicates}}",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Dataset & AIC Selection Rate & BIC Selection Rate \\",
        r"\midrule",
    ]
    part = summary.set_index("Scenario") if not summary.empty else pd.DataFrame()
    for sc in SCENARIOS:
        if not part.empty and sc.key in part.index:
            row = part.loc[sc.key]
            lines.append(f"{sc.label} & {fmt_rate(row['aic_selection_rate'])} & {fmt_rate(row['bic_selection_rate'])} \\\\")
        else:
            lines.append(f"{sc.label} & -- & -- \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:sim_selection}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_lrt_table(summary: pd.DataFrame, path: Path, reps_caption: int) -> None:
    lines = [
        r"\begin{table}[!h]",
        rf"\caption{{Likelihood ratio test results: rejection frequency of restricted models against the full ZIP--ZINB model out of {reps_caption} simulation replicates}}",
        r"\centering",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Dataset & Poisson & NB & P+NB & ZIP & ZINB & Z+P+NB \\",
        r"\midrule",
    ]
    if summary.empty:
        part = pd.DataFrame()
    else:
        part = summary.pivot(index="Scenario", columns="RestrictedModel", values="rejection_rate")
    for sc in SCENARIOS:
        if not part.empty and sc.key in part.index:
            vals = " & ".join(fmt_rate(part.loc[sc.key].get(m, np.nan)) for m in LRT_MODELS)
        else:
            vals = " & ".join("--" for _ in LRT_MODELS)
        lines.append(f"{sc.label} & {vals} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:sim_lrt}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_and_write(out_dir: Path, n_caption: int, reps_caption: int) -> None:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    yparams = existing_or_empty(out_dir / "raw_yparams_full.csv")
    coefs = existing_or_empty(out_dir / "raw_coefs_full.csv")
    selection = existing_or_empty(out_dir / "raw_model_selection.csv")
    lrt = existing_or_empty(out_dir / "raw_lrt.csv")
    runtime = existing_or_empty(out_dir / "raw_runtime.csv")
    logscore = existing_or_empty(out_dir / "raw_test_log_scores.csv")
    coverage = existing_or_empty(out_dir / "raw_coverage_full.csv")

    y_summary = summarise_yparams(yparams)
    c_summary = summarise_coefs(coefs)
    s_summary = summarise_selection(selection)
    l_summary = summarise_lrt(lrt)
    r_summary = summarise_runtime(runtime)
    oos_summary = summarise_logscore(logscore)
    cov_summary = summarise_coverage(coverage)

    y_summary.to_csv(tables_dir / "simulation_results_yparams.csv", index=False)
    c_summary.to_csv(tables_dir / "simulation_results_coefficients.csv", index=False)
    s_summary.to_csv(tables_dir / "simulation_selection_rates.csv", index=False)
    l_summary.to_csv(tables_dir / "simulation_lrt_rejection_rates.csv", index=False)
    r_summary.to_csv(tables_dir / "simulation_runtime_summary.csv", index=False)
    oos_summary.to_csv(tables_dir / "simulation_test_log_scores.csv", index=False)
    cov_summary.to_csv(tables_dir / "simulation_coverage_summary.csv", index=False)

    write_yparam_table(y_summary, tables_dir / "simulation_results_yparams.tex", n_caption, reps_caption)
    write_coef_table(c_summary, tables_dir / "simulation_results_coefficients.tex", reps_caption)
    write_selection_table(s_summary, tables_dir / "simulation_selection_rates.tex", reps_caption)
    write_lrt_table(l_summary, tables_dir / "simulation_lrt_rejection_rates.tex", reps_caption)

    manifest = [
        "Analytical simulation tables",
        "========================================",
        "",
        "LaTeX fragments:",
        "- simulation_results_yparams.tex        (label tab:sim)",
        "- simulation_results_coefficients.tex   (label tab:sim_more)",
        "- simulation_selection_rates.tex        (label tab:sim_selection)",
        "- simulation_lrt_rejection_rates.tex    (label tab:sim_lrt)",
        "- consistency_figure.tex                (label fig:consistency_s3_rmse)",
        "- fig_consistency_D1_nb2_identity.png",
        "",
        "Note: consistency_figure.tex uses a bare image name; copy the PNG next",
        "to any consuming TeX file or adjust \\graphicspath / the \\includegraphics path.",
        "",
        "CSV summaries:",
        "- simulation_results_yparams.csv",
        "- simulation_results_coefficients.csv",
        "- simulation_selection_rates.csv",
        "- simulation_lrt_rejection_rates.csv",
        "- simulation_runtime_summary.csv",
        "- simulation_test_log_scores.csv",
        "- simulation_coverage_summary.csv",
        "",
        "Raw/checkpoint CSVs live one directory above this tables folder.",
        "",
    ]
    (tables_dir / "README_tables.txt").write_text("\n".join(manifest), encoding="utf-8")


def read_completed_consistency(path: Path) -> set[tuple[int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return {(int(row.n), int(row.Rep)) for row in df.itertuples(index=False)}


def consistency_seed(n: int, rep: int, global_seed: int) -> int:
    return int((global_seed + 5_000_003 + n * 9176 + rep * 104_729) % (2**32 - 1))


def run_one_consistency_task(task: dict[str, object]) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sc = SCENARIO_BY_KEY["D1"]
    n = int(task["n"])
    rep = int(task["rep"])
    seed = int(task["seed"])
    n_starts = int(task["n_starts"])
    maxiter = int(task["maxiter"])
    use_true_starts = bool(task["use_true_starts"])
    t0 = time.time()

    data = generate_dataset(sc, n, seed)
    y = np.asarray(data["y"], dtype=float)
    X = np.asarray(data["X"], dtype=float)
    Z = np.asarray(data["Z"], dtype=float)
    extra = [truth_start(FULL_MODEL, sc)] if use_true_starts else []
    fit = fit_model(
        FULL_MODEL,
        y,
        X,
        Z,
        link=LINK,
        n_starts=n_starts,
        maxiter=maxiter,
        seed=seed + 2718,
        extra_starts=extra,
        compute_covariance=False,
    )
    true = true_coefs(sc)
    est = estimated_coefs(fit.params)
    rows = []
    for coef, latex in COEF_ORDER:
        rows.append(
            {
                "Scenario": sc.key,
                "Dataset": sc.label,
                "n": n,
                "Rep": rep,
                "Coefficient": coef,
                "CoefficientLatex": latex,
                "true": true[coef],
                "estimate": est[coef],
                "success": bool(fit.success),
                "Log-Likelihood": fit.llf,
                "AIC": fit.aic,
                "BIC": fit.bic,
            }
        )
    return {
        "n": n,
        "Rep": rep,
        "elapsed_sec": time.time() - t0,
        "rows": rows,
        "failure_rows": [],
    }


def summarise_consistency(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["n", "Coefficient", "CoefficientLatex", "true", "rmse", "n_success"])
    rows = []
    for (n, coef, latex), group in df.groupby(["n", "Coefficient", "CoefficientLatex"], sort=False):
        good = group[np.isfinite(group["estimate"].astype(float))]
        if good.empty:
            rmse = np.nan
            true = np.nan
        else:
            est = good["estimate"].astype(float)
            tru = good["true"].astype(float)
            rmse = float(np.sqrt(np.mean((est - tru) ** 2)))
            true = float(tru.mean())
        rows.append(
            {
                "n": int(n),
                "Coefficient": coef,
                "CoefficientLatex": latex,
                "true": true,
                "rmse": rmse,
                "n_success": int(len(good)),
            }
        )
    out = pd.DataFrame(rows)
    out["CoefOrder"] = out["Coefficient"].map({name: i for i, (name, _) in enumerate(COEF_ORDER)})
    return out.sort_values(["n", "CoefOrder"]).drop(columns=["CoefOrder"]).reset_index(drop=True)


def plot_consistency(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), constrained_layout=True)
    axes = axes.ravel()
    for ax, (coef, latex) in zip(axes, COEF_ORDER):
        sub = summary[summary["Coefficient"] == coef].sort_values("n")
        ax.plot(sub["n"], sub["rmse"], marker="o", linewidth=1.8)
        ax.set_title(latex, fontsize=13)
        ax.set_xlabel(r"Sample size $n$")
        ax.set_ylabel("RMSE")
        positive = sub["rmse"].to_numpy(dtype=float)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if positive.size:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    for k in range(len(COEF_ORDER), len(axes)):
        axes[k].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_consistency_figure_tex(path: Path, image_name: str, reps_caption: int) -> None:
    lines = [
        r"\begin{figure}[H]",
        r"    \centering",
        rf"    \includegraphics[width=\textwidth]{{{image_name}}}",
        rf"    \caption{{Consistency study for Dataset~1: empirical root mean squared error (RMSE) of regression coefficient estimates as a function of sample size $n$. Results are based on {reps_caption} repeated simulation replicates for each $n$.}}",
        r"    \label{fig:consistency_s3_rmse}",
        r"\end{figure}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_consistency_and_write(out_dir: Path, reps_caption: int) -> None:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw = existing_or_empty(out_dir / "raw_consistency_coefs.csv")
    summary = summarise_consistency(raw)
    summary.to_csv(tables_dir / "consistency_rmse.csv", index=False)
    image_name = "fig_consistency_D1_nb2_identity.png"
    if not summary.empty:
        plot_consistency(summary, tables_dir / image_name)
    write_consistency_figure_tex(tables_dir / "consistency_figure.tex", image_name, reps_caption)


def run_consistency_tasks(
    *,
    out_dir: Path,
    n_grid: list[int],
    reps: int,
    n_jobs: int,
    n_starts: int,
    maxiter: int,
    seed: int,
    use_true_starts: bool,
    no_resume: bool,
) -> None:
    completed_path = out_dir / "completed_consistency.csv"
    completed = set() if no_resume else read_completed_consistency(completed_path)
    tasks = []
    for n in n_grid:
        for rep in range(reps):
            if (int(n), int(rep)) in completed:
                continue
            tasks.append(
                {
                    "n": int(n),
                    "rep": int(rep),
                    "seed": consistency_seed(int(n), int(rep), seed),
                    "n_starts": int(n_starts),
                    "maxiter": int(maxiter),
                    "use_true_starts": bool(use_true_starts),
                }
            )

    print(f"Consistency n-grid: {', '.join(map(str, n_grid))}")
    print(f"Consistency reps per n: {reps}")
    print(f"Pending consistency tasks: {len(tasks)}")
    if completed:
        print(f"Already completed consistency tasks skipped: {len(completed)}")

    coef_cols = [
        "Scenario",
        "Dataset",
        "n",
        "Rep",
        "Coefficient",
        "CoefficientLatex",
        "true",
        "estimate",
        "success",
        "Log-Likelihood",
        "AIC",
        "BIC",
    ]
    failure_cols = ["n", "Rep", "Error"]
    completed_cols = ["n", "Rep", "elapsed_sec"]

    if tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(run_one_consistency_task, task): task for task in tasks}
            with tqdm(total=len(futures), desc="Consistency", unit="fit", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failure = {"n": task["n"], "Rep": task["rep"], "Error": repr(exc)}
                        append_rows(out_dir / "failures_consistency.csv", [failure], failure_cols)
                        pbar.update(1)
                        pbar.set_postfix_str(f"failed n={task['n']} rep={task['rep']}")
                        continue
                    append_rows(out_dir / "raw_consistency_coefs.csv", result["rows"], coef_cols)
                    append_rows(out_dir / "failures_consistency.csv", result["failure_rows"], failure_cols)
                    append_rows(
                        completed_path,
                        [{"n": result["n"], "Rep": result["Rep"], "elapsed_sec": result["elapsed_sec"]}],
                        completed_cols,
                    )
                    pbar.update(1)
                    pbar.set_postfix_str(f"n={result['n']} rep={result['Rep']}")

    print("Aggregating consistency output and writing plot...")
    aggregate_consistency_and_write(out_dir, reps_caption=reps)


def parse_scenarios(value: str) -> list[Scenario]:
    if value.strip().lower() in {"all", "*"}:
        return list(SCENARIOS)
    keys = [v.strip().upper() for v in value.split(",") if v.strip()]
    selected = []
    for key in keys:
        if key not in SCENARIO_BY_KEY:
            raise ValueError(f"Unknown scenario {key!r}; choose from D1,D2,D3,D4 or all")
        selected.append(SCENARIO_BY_KEY[key])
    return selected


def resolve_jobs(value: str) -> int:
    if value.lower() == "auto":
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1 if cpu > 1 else 1)
    jobs = int(value)
    return max(1, jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full analytical ZIP--ZINB simulations.")
    parser.add_argument("--reps", type=int, default=100, help="Monte Carlo replicates per scenario.")
    parser.add_argument("--n", type=int, default=None, help="Override per-scenario sample size; default is 20000.")
    parser.add_argument("--scenarios", default="all", help="Comma-separated scenario keys, e.g. D1,D3, or all.")
    parser.add_argument("--n-jobs", default="auto", help="Parallel workers: integer or auto. Default auto=cpu-1.")
    parser.add_argument("--n-starts", type=int, default=8, help="Optimizer starts per fitted model.")
    parser.add_argument("--maxiter", type=int, default=2000, help="SLSQP maxiter per optimizer start.")
    parser.add_argument("--seed", type=int, default=20260710, help="Global seed offset.")
    parser.add_argument("--alpha", type=float, default=0.05, help="LRT significance level.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "sim_analytical" / "full")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed scenario/rep pairs.")
    parser.add_argument("--no-true-starts", action="store_true", help="Do not seed optimizers with DGP parameters.")
    parser.add_argument("--n-test", type=int, default=0, help="If positive, generate this many fresh test observations per replicate and report out-of-sample log scores.")
    parser.add_argument("--compute-coverage", action="store_true", help="Compute observed-information standard errors for the full model and report empirical 95%% coverage.")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip the Dataset 1 consistency plot run.")
    parser.add_argument(
        "--consistency-reps",
        type=int,
        default=100,
        help="Consistency replicates per sample size.",
    )
    parser.add_argument(
        "--consistency-n-grid",
        default="1000,5000,10000,15000,25000",
        help="Comma-separated sample sizes for consistency plot.",
    )
    parser.add_argument(
        "--consistency-n-starts",
        type=int,
        default=None,
        help="Optimizer starts for consistency full-model fits; defaults to --n-starts.",
    )
    args = parser.parse_args()

    selected = parse_scenarios(args.scenarios)
    n_jobs = resolve_jobs(str(args.n_jobs))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_path = out_dir / "completed_reps.csv"
    completed = set() if args.no_resume else read_completed(completed_path)

    tasks = []
    for sc in selected:
        n = int(args.n if args.n is not None else sc.n)
        for rep in range(args.reps):
            if (sc.key, rep) in completed:
                continue
            tasks.append(
                {
                    "scenario": sc.key,
                    "rep": rep,
                    "n": n,
                    "seed": rep_seed(sc, rep, args.seed),
                    "n_starts": args.n_starts,
                    "maxiter": args.maxiter,
                    "alpha_level": args.alpha,
                    "use_true_starts": not args.no_true_starts,
                    "compute_coverage": bool(args.compute_coverage),
                    "n_test": int(args.n_test),
                }
            )

    n_caption = int(args.n if args.n is not None else selected[0].n)
    reps_caption = int(args.reps)

    print(f"Output directory: {out_dir}")
    print(f"Scenarios: {', '.join(sc.key for sc in selected)}")
    print(f"Replicates per scenario: {args.reps}")
    print(f"Sample size: {n_caption}")
    print(f"Parallel workers: {n_jobs}")
    print(f"Pending scenario/rep tasks: {len(tasks)}")
    if completed:
        print(f"Already completed and skipped: {len(completed)}")

    raw_selection_cols = [
        "Scenario",
        "DatasetLabel",
        "Rep",
        "Dataset",
        "Model",
        "Link",
        "Log-Likelihood",
        "AIC",
        "BIC",
        "nobs",
        "n_params",
        "success",
        "nit",
        "grad_norm_inf",
        "cov_status",
        "min_info_eig",
        "info_rank",
        "message",
        "fit_elapsed_sec",
    ]
    raw_yparam_cols = ["Scenario", "Dataset", "Rep", "Parameter", "true", "estimate"]
    raw_coef_cols = ["Scenario", "Dataset", "Rep", "Coefficient", "CoefficientLatex", "true", "estimate"]
    raw_param_cols = ["Scenario", "Dataset", "Rep", "Model", "Index", "Value"]
    raw_runtime_cols = ["Scenario", "Dataset", "Rep", "Model", "fit_elapsed_sec", "success", "nit", "grad_norm_inf"]
    raw_logscore_cols = ["Scenario", "Dataset", "Rep", "Model", "n_test", "test_loglik", "mean_test_log_score"]
    raw_coverage_cols = [
        "Scenario",
        "Dataset",
        "Rep",
        "Parameter",
        "Scale",
        "true",
        "estimate",
        "se",
        "lower95",
        "upper95",
        "covered",
        "CovarianceStatus",
    ]
    raw_lrt_cols = [
        "Scenario",
        "Dataset",
        "Rep",
        "RestrictedModel",
        "FullModel",
        "RestrictedLogLik",
        "FullLogLik",
        "LRT",
        "df",
        "boundary_dim",
        "p_value",
        "reject",
        "method",
    ]
    failure_cols = ["Scenario", "Dataset", "Rep", "Model", "Error"]
    completed_cols = ["Scenario", "Dataset", "Rep", "elapsed_sec"]

    if tasks:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(run_one_task, task): task for task in tasks}
            with tqdm(total=len(futures), desc="Simulation reps", unit="rep", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failure = {
                            "Scenario": task["scenario"],
                            "Dataset": SCENARIO_BY_KEY[str(task["scenario"])].label,
                            "Rep": task["rep"],
                            "Model": "__worker__",
                            "Error": repr(exc),
                        }
                        append_rows(out_dir / "failures.csv", [failure], failure_cols)
                        pbar.update(1)
                        pbar.set_postfix_str(f"failed {task['scenario']}:{task['rep']}")
                        continue

                    append_rows(out_dir / "raw_model_selection.csv", result["selection_rows"], raw_selection_cols)
                    append_rows(out_dir / "raw_yparams_full.csv", result["yparam_rows"], raw_yparam_cols)
                    append_rows(out_dir / "raw_coefs_full.csv", result["coef_rows"], raw_coef_cols)
                    append_rows(out_dir / "raw_params_full.csv", result["param_rows"], raw_param_cols)
                    append_rows(out_dir / "raw_runtime.csv", result["runtime_rows"], raw_runtime_cols)
                    append_rows(out_dir / "raw_test_log_scores.csv", result["logscore_rows"], raw_logscore_cols)
                    append_rows(out_dir / "raw_coverage_full.csv", result["coverage_rows"], raw_coverage_cols)
                    append_rows(out_dir / "raw_lrt.csv", result["lrt_rows"], raw_lrt_cols)
                    append_rows(out_dir / "failures.csv", result["failure_rows"], failure_cols)
                    append_rows(
                        completed_path,
                        [
                            {
                                "Scenario": result["Scenario"],
                                "Dataset": result["Dataset"],
                                "Rep": result["Rep"],
                                "elapsed_sec": result["elapsed_sec"],
                            }
                        ],
                        completed_cols,
                    )
                    pbar.update(1)
                    pbar.set_postfix_str(f"{result['Scenario']} rep {result['Rep']}")

    print("Aggregating current outputs and writing tables...")
    aggregate_and_write(out_dir, n_caption=n_caption, reps_caption=reps_caption)

    if not args.skip_consistency:
        n_grid = [int(x.strip()) for x in str(args.consistency_n_grid).split(",") if x.strip()]
        consistency_starts = int(args.consistency_n_starts or args.n_starts)
        run_consistency_tasks(
            out_dir=out_dir,
            n_grid=n_grid,
            reps=int(args.consistency_reps),
            n_jobs=n_jobs,
            n_starts=consistency_starts,
            maxiter=int(args.maxiter),
            seed=int(args.seed),
            use_true_starts=not args.no_true_starts,
            no_resume=bool(args.no_resume),
        )

    print(f"Done. Tables are in: {out_dir / 'tables'}")


if __name__ == "__main__":
    main()
