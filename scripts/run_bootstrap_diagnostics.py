"""Robustness checks for the analytical ZIP--ZINB simulation study.

This script is intentionally separate from the main simulation pipeline.
It is meant for quick pre-submission stress checks:

* parametric-bootstrap LRT calibration for selected nested models;
* fitted-frequency/rootogram data for the full model;
* observed versus fitted zero probabilities by dose bin;
* tail diagnostics;
* randomized quantile residual summaries;
* out-of-sample log scores;
* runtime summaries.

Run from the independent project root, for example:

    python scripts/run_bootstrap_diagnostics.py \
        --scenarios D3 --reps 10 --bootstrap-reps 49 --n-jobs auto
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/zip_zinb_matplotlib")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.stats import norm
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analytic_core import nb2_logpmf, poisson_logpmf  # noqa: E402
from analytic_fitters import MODEL_ORDER, fit_model  # noqa: E402
from run_simulation_analytical_full import (  # noqa: E402
    FULL_MODEL,
    LINK,
    LRT_MODELS,
    SCENARIO_BY_KEY,
    SCENARIOS,
    Scenario,
    estimated_coefs,
    estimated_yparams,
    generate_dataset,
    loglik_for_fit,
    rep_seed,
    simulate_response_given_design,
    truth_start,
    true_coefs,
    true_yparams,
    unpack_full,
    warm_starts,
)


EPS = 1e-12


def resolve_jobs(value: str) -> int:
    if value.lower() == "auto":
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1 if cpu > 1 else 1)
    return max(1, int(value))


def parse_scenarios(value: str) -> list[Scenario]:
    if value.strip().lower() in {"all", "*"}:
        return list(SCENARIOS)
    out: list[Scenario] = []
    for key in [v.strip().upper() for v in value.split(",") if v.strip()]:
        if key not in SCENARIO_BY_KEY:
            raise ValueError(f"Unknown scenario {key!r}; choose from D1,D2,D3,D4 or all")
        out.append(SCENARIO_BY_KEY[key])
    return out


def parse_models(value: str) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(LRT_MODELS)
    models = [v.strip() for v in value.split(",") if v.strip()]
    bad = [m for m in models if m not in LRT_MODELS]
    if bad:
        raise ValueError(f"Unknown restricted model(s): {bad}; choose from {LRT_MODELS}")
    return models


def append_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def inv_mean(X: np.ndarray, beta: np.ndarray, link: str) -> np.ndarray:
    eta = np.asarray(X @ beta, dtype=float)
    if link == "log":
        return np.exp(np.clip(eta, -30.0, 30.0))
    if link == "identity":
        return np.clip(eta, EPS, None)
    raise ValueError(link)


def unpack_model(model: str, params: np.ndarray, p: int, q: int) -> dict[str, object]:
    params = np.asarray(params, dtype=float)
    if model == "Poisson":
        return {"beta_p": params[:p]}
    if model == "NB":
        return {"beta_nb": params[:p], "alpha": float(np.exp(params[p]))}
    if model == "ZIP":
        return {"beta_p": params[:p], "gamma_p": params[p : p + q]}
    if model == "ZINB":
        return {"beta_nb": params[:p], "gamma_nb": params[p : p + q], "alpha": float(np.exp(params[p + q]))}
    if model == "P+NB":
        return {
            "rho": float(expit(params[0])),
            "beta_p": params[1 : 1 + p],
            "beta_nb": params[1 + p : 1 + 2 * p],
            "alpha": float(np.exp(params[-1])),
        }
    if model == "Z+P+NB":
        return {
            "rho": float(expit(params[0])),
            "beta_p": params[1 : 1 + p],
            "beta_nb": params[1 + p : 1 + 2 * p],
            "gamma": params[1 + 2 * p : 1 + 2 * p + q],
            "alpha": float(np.exp(params[-1])),
        }
    if model == "ZIP--ZINB":
        full = unpack_full(params)
        return {
            "rho": float(full["rho"]),
            "beta_p": np.asarray(full["beta_z"], dtype=float),
            "gamma_p": np.asarray(full["gamma_z"], dtype=float),
            "beta_nb": np.asarray(full["beta_nb"], dtype=float),
            "gamma_nb": np.asarray(full["gamma_nb"], dtype=float),
            "alpha": float(full["alpha"]),
        }
    raise ValueError(model)


def poisson_pmf_grid(y_grid: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.exp(poisson_logpmf(y_grid[:, None], mu[None, :]))


def nb_pmf_grid(y_grid: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    return np.exp(nb2_logpmf(y_grid[:, None], mu[None, :], alpha))


def pmf_grid(model: str, params: np.ndarray, X: np.ndarray, Z: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    p, q = X.shape[1], Z.shape[1]
    par = unpack_model(model, params, p, q)
    y_grid = np.asarray(y_grid, dtype=float)
    is_zero = (y_grid == 0.0)[:, None]

    if model == "Poisson":
        return poisson_pmf_grid(y_grid, inv_mean(X, par["beta_p"], LINK))
    if model == "NB":
        return nb_pmf_grid(y_grid, inv_mean(X, par["beta_nb"], LINK), float(par["alpha"]))
    if model == "ZIP":
        mu = inv_mean(X, par["beta_p"], LINK)
        pi = expit(Z @ par["gamma_p"])
        pois = poisson_pmf_grid(y_grid, mu)
        return np.where(is_zero, pi[None, :] + (1.0 - pi)[None, :] * pois, (1.0 - pi)[None, :] * pois)
    if model == "ZINB":
        mu = inv_mean(X, par["beta_nb"], LINK)
        omega = expit(Z @ par["gamma_nb"])
        nb = nb_pmf_grid(y_grid, mu, float(par["alpha"]))
        return np.where(is_zero, omega[None, :] + (1.0 - omega)[None, :] * nb, (1.0 - omega)[None, :] * nb)
    if model == "P+NB":
        rho = float(par["rho"])
        pois = poisson_pmf_grid(y_grid, inv_mean(X, par["beta_p"], LINK))
        nb = nb_pmf_grid(y_grid, inv_mean(X, par["beta_nb"], LINK), float(par["alpha"]))
        return rho * pois + (1.0 - rho) * nb
    if model == "Z+P+NB":
        rho = float(par["rho"])
        gamma = np.asarray(par["gamma"], dtype=float)
        zi = expit(Z @ gamma)
        pois = poisson_pmf_grid(y_grid, inv_mean(X, par["beta_p"], LINK))
        nb = nb_pmf_grid(y_grid, inv_mean(X, par["beta_nb"], LINK), float(par["alpha"]))
        count = rho * pois + (1.0 - rho) * nb
        return np.where(is_zero, zi[None, :] + (1.0 - zi)[None, :] * count, (1.0 - zi)[None, :] * count)
    if model == "ZIP--ZINB":
        rho = float(par["rho"])
        pi = expit(Z @ par["gamma_p"])
        omega = expit(Z @ par["gamma_nb"])
        pois = poisson_pmf_grid(y_grid, inv_mean(X, par["beta_p"], LINK))
        nb = nb_pmf_grid(y_grid, inv_mean(X, par["beta_nb"], LINK), float(par["alpha"]))
        zip_part = np.where(is_zero, pi[None, :] + (1.0 - pi)[None, :] * pois, (1.0 - pi)[None, :] * pois)
        zinb_part = np.where(is_zero, omega[None, :] + (1.0 - omega)[None, :] * nb, (1.0 - omega)[None, :] * nb)
        return rho * zip_part + (1.0 - rho) * zinb_part
    raise ValueError(model)


def logpmf_observed(model: str, params: np.ndarray, y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    probs = pmf_grid(model, params, X, Z, np.asarray(y, dtype=float))
    return np.log(np.clip(np.diag(probs), EPS, None))


def zero_probability(model: str, params: np.ndarray, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return pmf_grid(model, params, X, Z, np.array([0.0]))[0]


def simulate_from_model(model: str, params: np.ndarray, X: np.ndarray, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p, q = X.shape[1], Z.shape[1]
    par = unpack_model(model, params, p, q)
    n = X.shape[0]

    def draw_pois(beta: np.ndarray) -> np.ndarray:
        return rng.poisson(inv_mean(X, beta, LINK))

    def draw_nb(beta: np.ndarray, alpha: float) -> np.ndarray:
        mu = inv_mean(X, beta, LINK)
        r = 1.0 / alpha
        prob = r / (r + mu)
        return rng.negative_binomial(r, prob)

    if model == "Poisson":
        return draw_pois(par["beta_p"]).astype(float)
    if model == "NB":
        return draw_nb(par["beta_nb"], float(par["alpha"])).astype(float)
    if model == "ZIP":
        y = draw_pois(par["beta_p"])
        y[rng.random(n) < expit(Z @ par["gamma_p"])] = 0
        return y.astype(float)
    if model == "ZINB":
        y = draw_nb(par["beta_nb"], float(par["alpha"]))
        y[rng.random(n) < expit(Z @ par["gamma_nb"])] = 0
        return y.astype(float)
    if model == "P+NB":
        from_p = rng.random(n) < float(par["rho"])
        y = np.empty(n, dtype=float)
        y[from_p] = draw_pois(par["beta_p"])[from_p]
        y[~from_p] = draw_nb(par["beta_nb"], float(par["alpha"]))[~from_p]
        return y
    if model == "Z+P+NB":
        from_p = rng.random(n) < float(par["rho"])
        y_p = draw_pois(par["beta_p"])
        y_n = draw_nb(par["beta_nb"], float(par["alpha"]))
        y = np.where(from_p, y_p, y_n)
        y[rng.random(n) < expit(Z @ par["gamma"])] = 0
        return y.astype(float)
    if model == "ZIP--ZINB":
        from_zip = rng.random(n) < float(par["rho"])
        y_p = draw_pois(par["beta_p"])
        y_p[rng.random(n) < expit(Z @ par["gamma_p"])] = 0
        y_n = draw_nb(par["beta_nb"], float(par["alpha"]))
        y_n[rng.random(n) < expit(Z @ par["gamma_nb"])] = 0
        return np.where(from_zip, y_p, y_n).astype(float)
    raise ValueError(model)


def full_start_from_restricted(model: str, params: np.ndarray, p: int, q: int) -> np.ndarray:
    par = unpack_model(model, params, p, q)
    gamma0 = np.zeros(q)
    log_alpha0 = np.log(float(par.get("alpha", 0.5)))
    if model == "Poisson":
        return np.r_[0.0, par["beta_p"], gamma0, par["beta_p"], gamma0, log_alpha0]
    if model == "NB":
        return np.r_[0.0, par["beta_nb"], gamma0, par["beta_nb"], gamma0, log_alpha0]
    if model == "ZIP":
        return np.r_[0.0, par["beta_p"], par["gamma_p"], par["beta_p"], par["gamma_p"], log_alpha0]
    if model == "ZINB":
        return np.r_[0.0, par["beta_nb"], par["gamma_nb"], par["beta_nb"], par["gamma_nb"], log_alpha0]
    if model == "P+NB":
        return np.r_[params[0], par["beta_p"], gamma0, par["beta_nb"], gamma0, log_alpha0]
    if model == "Z+P+NB":
        return np.r_[params[0], par["beta_p"], par["gamma"], par["beta_nb"], par["gamma"], log_alpha0]
    raise ValueError(model)


def fit_all_models(sc: Scenario, y: np.ndarray, X: np.ndarray, Z: np.ndarray, *, seed: int, n_starts: int, maxiter: int) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    fits: dict[str, object] = {}
    failures: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for i, model in enumerate(MODEL_ORDER):
        t0 = time.time()
        try:
            fit = fit_model(
                model,
                y,
                X,
                Z,
                link=LINK,
                n_starts=n_starts,
                maxiter=maxiter,
                seed=seed + 1009 * (i + 1),
                extra_starts=warm_starts(model, fits, sc, True),
                compute_covariance=False,
            )
            fits[model] = fit
            runtime_rows.append(
                {
                    "Model": model,
                    "fit_elapsed_sec": time.time() - t0,
                    "success": bool(fit.success),
                    "nit": int(fit.nit),
                    "llf": float(fit.llf),
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                }
            )
        except Exception as exc:
            failures.append({"Model": model, "Error": repr(exc)})
    return fits, runtime_rows, failures


def bootstrap_lrt_rows(
    sc: Scenario,
    rep: int,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    fits: dict[str, object],
    models: list[str],
    *,
    bootstrap_reps: int,
    n_starts: int,
    maxiter: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    if FULL_MODEL not in fits:
        return rows, failures
    full = fits[FULL_MODEL]
    p, q = X.shape[1], Z.shape[1]
    rng = np.random.default_rng(seed)
    for model in models:
        if model not in fits:
            continue
        restricted = fits[model]
        observed = max(0.0, 2.0 * (full.llf - restricted.llf))
        boot_stats: list[float] = []
        t0 = time.time()
        for b in range(bootstrap_reps):
            try:
                yb = simulate_from_model(model, restricted.params, X, Z, rng)
                fit_r = fit_model(
                    model,
                    yb,
                    X,
                    Z,
                    link=LINK,
                    n_starts=n_starts,
                    maxiter=maxiter,
                    seed=int(rng.integers(1, 2**31 - 1)),
                    extra_starts=[restricted.params],
                    compute_covariance=False,
                )
                fit_f = fit_model(
                    FULL_MODEL,
                    yb,
                    X,
                    Z,
                    link=LINK,
                    n_starts=n_starts,
                    maxiter=maxiter,
                    seed=int(rng.integers(1, 2**31 - 1)),
                    extra_starts=[full_start_from_restricted(model, fit_r.params, p, q), full.params],
                    compute_covariance=False,
                )
                boot_stats.append(max(0.0, 2.0 * (fit_f.llf - fit_r.llf)))
            except Exception as exc:
                failures.append(
                    {
                        "Scenario": sc.key,
                        "Dataset": sc.label,
                        "Rep": rep,
                        "RestrictedModel": model,
                        "BootstrapRep": b,
                        "Error": repr(exc),
                    }
                )
        boot = np.asarray(boot_stats, dtype=float)
        p_boot = np.nan if boot.size == 0 else float((1.0 + np.sum(boot >= observed)) / (boot.size + 1.0))
        rows.append(
            {
                "Scenario": sc.key,
                "Dataset": sc.label,
                "Rep": rep,
                "RestrictedModel": model,
                "ObservedLRT": observed,
                "BootstrapPValue": p_boot,
                "BootstrapRepsRequested": bootstrap_reps,
                "BootstrapRepsSucceeded": int(boot.size),
                "BootstrapMeanLRT": float(np.mean(boot)) if boot.size else np.nan,
                "BootstrapQ95": float(np.quantile(boot, 0.95)) if boot.size else np.nan,
                "ElapsedSec": time.time() - t0,
            }
        )
    return rows, failures


def diagnostic_rows(sc: Scenario, rep: int, y: np.ndarray, X: np.ndarray, Z: np.ndarray, full_params: np.ndarray) -> dict[str, list[dict[str, object]]]:
    max_observed = int(np.nanmax(y)) if len(y) else 0
    max_count = int(min(max(15, max_observed), 60))
    y_grid = np.arange(max_count + 1, dtype=float)
    probs = pmf_grid(FULL_MODEL, full_params, X, Z, y_grid)
    expected = probs.sum(axis=1)
    observed = np.bincount(np.asarray(y, dtype=int), minlength=max_count + 1)[: max_count + 1].astype(float)
    tail_observed = float(np.sum(y > max_count))
    tail_expected = float(np.sum(np.clip(1.0 - probs.sum(axis=0), 0.0, 1.0)))

    freq_rows = []
    for yy, obs, exp in zip(y_grid.astype(int), observed, expected):
        freq_rows.append(
            {
                "Scenario": sc.key,
                "Dataset": sc.label,
                "Rep": rep,
                "Count": int(yy),
                "Observed": obs,
                "Expected": exp,
                "RootogramDiff": np.sqrt(obs) - np.sqrt(max(exp, 0.0)),
            }
        )
    freq_rows.append(
        {
            "Scenario": sc.key,
            "Dataset": sc.label,
            "Rep": rep,
            "Count": f">{max_count}",
            "Observed": tail_observed,
            "Expected": tail_expected,
            "RootogramDiff": np.sqrt(tail_observed) - np.sqrt(max(tail_expected, 0.0)),
        }
    )

    dose = X[:, 1]
    bins = np.quantile(dose, np.linspace(0.0, 1.0, 11))
    bins = np.unique(bins)
    zprob = zero_probability(FULL_MODEL, full_params, X, Z)
    zero_rows = []
    if bins.size > 1:
        labels = np.digitize(dose, bins[1:-1], right=True)
        for b in range(bins.size - 1):
            mask = labels == b
            if not np.any(mask):
                continue
            zero_rows.append(
                {
                    "Scenario": sc.key,
                    "Dataset": sc.label,
                    "Rep": rep,
                    "Bin": b + 1,
                    "DoseMin": float(np.min(dose[mask])),
                    "DoseMax": float(np.max(dose[mask])),
                    "n": int(np.sum(mask)),
                    "ObservedZeroRate": float(np.mean(y[mask] == 0)),
                    "FittedZeroProbability": float(np.mean(zprob[mask])),
                }
            )

    tail_rows = []
    for threshold in [1, 2, 3, 5, 8, 10, 15, 20]:
        if threshold > max_count:
            continue
        expected_tail = float(np.sum(np.clip(1.0 - probs[:threshold, :].sum(axis=0), 0.0, 1.0)))
        observed_tail = float(np.sum(y >= threshold))
        tail_rows.append(
            {
                "Scenario": sc.key,
                "Dataset": sc.label,
                "Rep": rep,
                "Threshold": threshold,
                "ObservedTailCount": observed_tail,
                "ExpectedTailCount": expected_tail,
                "ObservedTailRate": observed_tail / len(y),
                "ExpectedTailRate": expected_tail / len(y),
            }
        )

    rng = np.random.default_rng(81_771 + rep)
    y_int = np.asarray(y, dtype=int)
    y_low = np.maximum(y_int - 1, 0)
    y_unique = np.unique(np.r_[y_low, y_int]).astype(float)
    cdf_grid = np.cumsum(pmf_grid(FULL_MODEL, full_params, X, Z, np.arange(0, max(max_count, int(np.max(y_int))) + 1)), axis=0)
    lower = np.zeros(len(y), dtype=float)
    upper = np.ones(len(y), dtype=float)
    within = y_int <= cdf_grid.shape[0] - 1
    upper[within] = cdf_grid[y_int[within], np.where(within)[0]]
    positive = within & (y_int > 0)
    lower[positive] = cdf_grid[y_int[positive] - 1, np.where(positive)[0]]
    u = lower + rng.random(len(y)) * np.maximum(upper - lower, EPS)
    resid = norm.ppf(np.clip(u, 1e-9, 1.0 - 1e-9))
    residual_rows = [
        {
            "Scenario": sc.key,
            "Dataset": sc.label,
            "Rep": rep,
            "n": len(y),
            "mean": float(np.mean(resid)),
            "sd": float(np.std(resid, ddof=1)),
            "q01": float(np.quantile(resid, 0.01)),
            "q05": float(np.quantile(resid, 0.05)),
            "q50": float(np.quantile(resid, 0.50)),
            "q95": float(np.quantile(resid, 0.95)),
            "q99": float(np.quantile(resid, 0.99)),
        }
    ]
    return {"freq": freq_rows, "zero": zero_rows, "tail": tail_rows, "residual": residual_rows}


def run_one(task: dict[str, object]) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sc = SCENARIO_BY_KEY[str(task["scenario"])]
    rep = int(task["rep"])
    seed = int(task["seed"])
    n = int(task["n"])
    n_starts = int(task["n_starts"])
    maxiter = int(task["maxiter"])
    n_test = int(task["n_test"])
    bootstrap_reps = int(task["bootstrap_reps"])
    bootstrap_models = list(task["bootstrap_models"])
    diagnostic = bool(task["diagnostic"])

    data = generate_dataset(sc, n, seed)
    y = np.asarray(data["y"], dtype=float)
    X = np.asarray(data["X"], dtype=float)
    Z = np.asarray(data["Z"], dtype=float)

    fits, runtime_rows, failures = fit_all_models(sc, y, X, Z, seed=seed, n_starts=n_starts, maxiter=maxiter)
    for row in runtime_rows:
        row.update({"Scenario": sc.key, "Dataset": sc.label, "Rep": rep})
    for row in failures:
        row.update({"Scenario": sc.key, "Dataset": sc.label, "Rep": rep})

    yparam_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    if FULL_MODEL in fits:
        y_true = true_yparams(sc, data)
        y_est = estimated_yparams(fits[FULL_MODEL].params, X, Z)
        for key, val in y_true.items():
            yparam_rows.append({"Scenario": sc.key, "Dataset": sc.label, "Rep": rep, "Parameter": key, "true": val, "estimate": y_est[key]})
        c_true = true_coefs(sc)
        c_est = estimated_coefs(fits[FULL_MODEL].params)
        for key, val in c_true.items():
            coef_rows.append({"Scenario": sc.key, "Dataset": sc.label, "Rep": rep, "Parameter": key, "true": val, "estimate": c_est[key]})

    logscore_rows: list[dict[str, object]] = []
    if n_test > 0:
        test_rng = np.random.default_rng(seed + 8_000_003)
        idx = test_rng.choice(np.arange(len(y)), size=n_test, replace=True)
        X_test = X[idx, :]
        Z_test = Z[idx, :]
        y_test, *_ = simulate_response_given_design(sc, X_test, Z_test, test_rng)
        for model, fit in fits.items():
            try:
                ll = loglik_for_fit(model, fit.params, y_test, X_test, Z_test)
                logscore_rows.append(
                    {
                        "Scenario": sc.key,
                        "Dataset": sc.label,
                        "Rep": rep,
                        "Model": model,
                        "n_test": n_test,
                        "test_loglik": ll,
                        "mean_test_log_score": ll / n_test,
                    }
                )
            except Exception as exc:
                failures.append({"Scenario": sc.key, "Dataset": sc.label, "Rep": rep, "Model": f"{model}__test_logscore", "Error": repr(exc)})

    diag = {"freq": [], "zero": [], "tail": [], "residual": []}
    if diagnostic and FULL_MODEL in fits:
        diag = diagnostic_rows(sc, rep, y, X, Z, fits[FULL_MODEL].params)

    boot_rows: list[dict[str, object]] = []
    boot_failures: list[dict[str, object]] = []
    if bootstrap_reps > 0 and FULL_MODEL in fits:
        boot_rows, boot_failures = bootstrap_lrt_rows(
            sc,
            rep,
            y,
            X,
            Z,
            fits,
            bootstrap_models,
            bootstrap_reps=bootstrap_reps,
            n_starts=max(1, min(3, n_starts)),
            maxiter=maxiter,
            seed=seed + 31_415,
        )

    return {
        "Scenario": sc.key,
        "Dataset": sc.label,
        "Rep": rep,
        "runtime_rows": runtime_rows,
        "failure_rows": failures + boot_failures,
        "yparam_rows": yparam_rows,
        "coef_rows": coef_rows,
        "logscore_rows": logscore_rows,
        "bootstrap_rows": boot_rows,
        "freq_rows": diag["freq"],
        "zero_rows": diag["zero"],
        "tail_rows": diag["tail"],
        "residual_rows": diag["residual"],
    }


def summarise_and_plot(out_dir: Path) -> None:
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    for name, group_cols, value_cols in [
        ("runtime", ["Scenario", "Dataset", "Model"], ["fit_elapsed_sec"]),
        ("test_log_scores", ["Scenario", "Dataset", "Model"], ["mean_test_log_score"]),
        ("bootstrap_lrt", ["Scenario", "Dataset", "RestrictedModel"], ["BootstrapPValue", "ObservedLRT", "BootstrapQ95"]),
    ]:
        path = out_dir / f"raw_{name}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        rows = []
        for keys, group in df.groupby(group_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            for col in value_cols:
                vals = group[col].astype(float)
                row[f"mean_{col}"] = float(vals.mean())
                row[f"sd_{col}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"n_{col}"] = int(vals.notna().sum())
            rows.append(row)
        pd.DataFrame(rows).to_csv(tables / f"summary_{name}.csv", index=False)

    freq_path = out_dir / "diagnostic_fitted_frequency.csv"
    if freq_path.exists() and freq_path.stat().st_size > 0:
        freq = pd.read_csv(freq_path)
        for (scenario, rep), sub in freq.groupby(["Scenario", "Rep"], sort=False):
            sub_plot = sub[sub["Count"].astype(str).str.startswith(">") == False].copy()
            counts = sub_plot["Count"].astype(int)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(counts - 0.2, sub_plot["Observed"], width=0.4, label="Observed", alpha=0.75)
            ax.bar(counts + 0.2, sub_plot["Expected"], width=0.4, label="Fitted", alpha=0.75)
            ax.set_xlabel("Count")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{scenario} rep {rep}: observed and fitted frequencies")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"fitted_frequency_{scenario}_rep{rep}.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.bar(counts, sub_plot["RootogramDiff"], alpha=0.8)
            ax.set_xlabel("Count")
            ax.set_ylabel(r"$\sqrt{O_y}-\sqrt{E_y}$")
            ax.set_title(f"{scenario} rep {rep}: hanging rootogram residuals")
            fig.tight_layout()
            fig.savefig(figures / f"rootogram_{scenario}_rep{rep}.png", dpi=180)
            plt.close(fig)

    zero_path = out_dir / "diagnostic_zero_probabilities.csv"
    if zero_path.exists() and zero_path.stat().st_size > 0:
        zero = pd.read_csv(zero_path)
        for (scenario, rep), sub in zero.groupby(["Scenario", "Rep"], sort=False):
            fig, ax = plt.subplots(figsize=(7, 4))
            mids = 0.5 * (sub["DoseMin"] + sub["DoseMax"])
            ax.plot(mids, sub["ObservedZeroRate"], marker="o", label="Observed zero rate")
            ax.plot(mids, sub["FittedZeroProbability"], marker="s", label="Fitted zero probability")
            ax.set_xlabel("Dose")
            ax.set_ylabel("Zero probability")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{scenario} rep {rep}: zero-probability calibration")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"zero_probability_{scenario}_rep{rep}.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bootstrap LRT and diagnostics for analytical ZIP--ZINB simulations.")
    parser.add_argument("--scenarios", default="D3")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-jobs", default="auto")
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--bootstrap-reps", type=int, default=49)
    parser.add_argument("--bootstrap-models", default="Z+P+NB,P+NB,ZIP,ZINB")
    parser.add_argument("--diagnostic-reps", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "robustness_checks")
    args = parser.parse_args()

    selected = parse_scenarios(args.scenarios)
    models = parse_models(args.bootstrap_models)
    n_jobs = resolve_jobs(str(args.n_jobs))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for sc in selected:
        n = int(args.n if args.n is not None else sc.n)
        for rep in range(args.reps):
            tasks.append(
                {
                    "scenario": sc.key,
                    "rep": rep,
                    "seed": rep_seed(sc, rep, args.seed),
                    "n": n,
                    "n_starts": args.n_starts,
                    "maxiter": args.maxiter,
                    "n_test": args.n_test,
                    "bootstrap_reps": args.bootstrap_reps,
                    "bootstrap_models": models,
                    "diagnostic": rep < args.diagnostic_reps,
                }
            )

    print(f"Output directory: {out_dir}")
    print(f"Scenarios: {', '.join(sc.key for sc in selected)}")
    print(f"Reps: {args.reps}; bootstrap reps: {args.bootstrap_reps}; models: {', '.join(models)}")
    print(f"Parallel workers: {n_jobs}; tasks: {len(tasks)}")

    cols = {
        "raw_runtime.csv": ["Scenario", "Dataset", "Rep", "Model", "fit_elapsed_sec", "success", "nit", "llf", "aic", "bic"],
        "failures.csv": ["Scenario", "Dataset", "Rep", "Model", "RestrictedModel", "BootstrapRep", "Error"],
        "raw_yparams.csv": ["Scenario", "Dataset", "Rep", "Parameter", "true", "estimate"],
        "raw_coefs.csv": ["Scenario", "Dataset", "Rep", "Parameter", "true", "estimate"],
        "raw_test_log_scores.csv": ["Scenario", "Dataset", "Rep", "Model", "n_test", "test_loglik", "mean_test_log_score"],
        "raw_bootstrap_lrt.csv": [
            "Scenario",
            "Dataset",
            "Rep",
            "RestrictedModel",
            "ObservedLRT",
            "BootstrapPValue",
            "BootstrapRepsRequested",
            "BootstrapRepsSucceeded",
            "BootstrapMeanLRT",
            "BootstrapQ95",
            "ElapsedSec",
        ],
        "diagnostic_fitted_frequency.csv": ["Scenario", "Dataset", "Rep", "Count", "Observed", "Expected", "RootogramDiff"],
        "diagnostic_zero_probabilities.csv": ["Scenario", "Dataset", "Rep", "Bin", "DoseMin", "DoseMax", "n", "ObservedZeroRate", "FittedZeroProbability"],
        "diagnostic_tail.csv": ["Scenario", "Dataset", "Rep", "Threshold", "ObservedTailCount", "ExpectedTailCount", "ObservedTailRate", "ExpectedTailRate"],
        "diagnostic_rqr_summary.csv": ["Scenario", "Dataset", "Rep", "n", "mean", "sd", "q01", "q05", "q50", "q95", "q99"],
    }

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        with tqdm(total=len(futures), desc="Robustness", unit="rep", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    sc = SCENARIO_BY_KEY[str(task["scenario"])]
                    append_rows(
                        out_dir / "failures.csv",
                        [{"Scenario": sc.key, "Dataset": sc.label, "Rep": task["rep"], "Model": "__worker__", "Error": repr(exc)}],
                        cols["failures.csv"],
                    )
                    pbar.update(1)
                    pbar.set_postfix_str(f"failed {task['scenario']}:{task['rep']}")
                    continue

                append_rows(out_dir / "raw_runtime.csv", result["runtime_rows"], cols["raw_runtime.csv"])
                append_rows(out_dir / "failures.csv", result["failure_rows"], cols["failures.csv"])
                append_rows(out_dir / "raw_yparams.csv", result["yparam_rows"], cols["raw_yparams.csv"])
                append_rows(out_dir / "raw_coefs.csv", result["coef_rows"], cols["raw_coefs.csv"])
                append_rows(out_dir / "raw_test_log_scores.csv", result["logscore_rows"], cols["raw_test_log_scores.csv"])
                append_rows(out_dir / "raw_bootstrap_lrt.csv", result["bootstrap_rows"], cols["raw_bootstrap_lrt.csv"])
                append_rows(out_dir / "diagnostic_fitted_frequency.csv", result["freq_rows"], cols["diagnostic_fitted_frequency.csv"])
                append_rows(out_dir / "diagnostic_zero_probabilities.csv", result["zero_rows"], cols["diagnostic_zero_probabilities.csv"])
                append_rows(out_dir / "diagnostic_tail.csv", result["tail_rows"], cols["diagnostic_tail.csv"])
                append_rows(out_dir / "diagnostic_rqr_summary.csv", result["residual_rows"], cols["diagnostic_rqr_summary.csv"])
                pbar.update(1)
                pbar.set_postfix_str(f"{result['Scenario']} rep {result['Rep']}")

    summarise_and_plot(out_dir)
    print(f"Done. Summaries/figures are in {out_dir}")


if __name__ == "__main__":
    main()
