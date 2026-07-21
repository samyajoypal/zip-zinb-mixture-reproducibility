"""Extended real-data diagnostics for the analytical ZIP--ZINB project.

This script is separate from the frozen real-data analysis.  It is meant for
the Statistics & Computing version, where we want additional checks:

* parametric-bootstrap LRTs against the full ZIP--ZINB model;
* fitted-frequency and hanging-rootogram data/figures;
* observed versus fitted zero probabilities by dose;
* tail diagnostics;
* randomized quantile residual summaries;
* K-fold out-of-sample log scores;
* fit/runtime summaries.

Examples from the independent project root:

    .venv/bin/python scripts/run_real_extended_diagnostics.py \
        --mode diagnostics --datasets all --links all --n-jobs auto

    .venv/bin/python scripts/run_real_extended_diagnostics.py \
        --mode bootstrap --datasets C2 --links identity \
        --bootstrap-models Z+P+NB --bootstrap-reps 99 --n-jobs auto

    .venv/bin/python scripts/run_real_extended_diagnostics.py \
        --mode cv --cv-folds 5 --n-jobs auto
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analytic_core import nb2_logpmf, poisson_logpmf  # noqa: E402
from analytic_fitters import LINKS, MODEL_ORDER, fit_model, result_row  # noqa: E402
from run_real_analytical import DATASETS, DatasetSpec, load_design, warm_starts  # noqa: E402


FULL_MODEL = "ZIP--ZINB"
LRT_MODELS = ("Poisson", "NB", "P+NB", "ZIP", "ZINB", "Z+P+NB")
EPS = 1e-12

DATASET_BY_KEY = {spec.key: spec for spec in DATASETS}


def resolve_jobs(value: str) -> int:
    if value.lower() == "auto":
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1 if cpu > 1 else 1)
    return max(1, int(value))


def parse_datasets(value: str) -> list[DatasetSpec]:
    if value.strip().lower() in {"all", "*"}:
        return list(DATASETS)
    out: list[DatasetSpec] = []
    for key in [v.strip() for v in value.split(",") if v.strip()]:
        if key not in DATASET_BY_KEY:
            raise ValueError(f"Unknown dataset {key!r}; choose from {list(DATASET_BY_KEY)} or all")
        out.append(DATASET_BY_KEY[key])
    return out


def parse_links(value: str) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(LINKS)
    out = [v.strip().lower() for v in value.split(",") if v.strip()]
    bad = [v for v in out if v not in LINKS]
    if bad:
        raise ValueError(f"Unknown link(s): {bad}; choose from {LINKS} or all")
    return out


def parse_models(value: str) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(LRT_MODELS)
    out = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in out if v not in LRT_MODELS]
    if bad:
        raise ValueError(f"Unknown restricted model(s): {bad}; choose from {LRT_MODELS} or all")
    return out


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
        return {
            "rho": float(expit(params[0])),
            "beta_p": params[1 : 1 + p],
            "gamma_p": params[1 + p : 1 + p + q],
            "beta_nb": params[1 + p + q : 1 + p + q + p],
            "gamma_nb": params[1 + p + q + p : 1 + p + q + p + q],
            "alpha": float(np.exp(params[-1])),
        }
    raise ValueError(model)


def poisson_pmf_grid(y_grid: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.exp(poisson_logpmf(y_grid[:, None], mu[None, :]))


def nb_pmf_grid(y_grid: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    return np.exp(nb2_logpmf(y_grid[:, None], mu[None, :], alpha))


def pmf_grid(model: str, params: np.ndarray, X: np.ndarray, Z: np.ndarray, y_grid: np.ndarray, link: str) -> np.ndarray:
    p, q = X.shape[1], Z.shape[1]
    par = unpack_model(model, params, p, q)
    y_grid = np.asarray(y_grid, dtype=float)
    is_zero = (y_grid == 0.0)[:, None]

    if model == "Poisson":
        return poisson_pmf_grid(y_grid, inv_mean(X, par["beta_p"], link))
    if model == "NB":
        return nb_pmf_grid(y_grid, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
    if model == "ZIP":
        mu = inv_mean(X, par["beta_p"], link)
        pi = expit(Z @ par["gamma_p"])
        pois = poisson_pmf_grid(y_grid, mu)
        return np.where(is_zero, pi[None, :] + (1.0 - pi)[None, :] * pois, (1.0 - pi)[None, :] * pois)
    if model == "ZINB":
        mu = inv_mean(X, par["beta_nb"], link)
        omega = expit(Z @ par["gamma_nb"])
        nb = nb_pmf_grid(y_grid, mu, float(par["alpha"]))
        return np.where(is_zero, omega[None, :] + (1.0 - omega)[None, :] * nb, (1.0 - omega)[None, :] * nb)
    if model == "P+NB":
        rho = float(par["rho"])
        pois = poisson_pmf_grid(y_grid, inv_mean(X, par["beta_p"], link))
        nb = nb_pmf_grid(y_grid, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
        return rho * pois + (1.0 - rho) * nb
    if model == "Z+P+NB":
        rho = float(par["rho"])
        gamma = np.asarray(par["gamma"], dtype=float)
        zi = expit(Z @ gamma)
        pois = poisson_pmf_grid(y_grid, inv_mean(X, par["beta_p"], link))
        nb = nb_pmf_grid(y_grid, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
        count = rho * pois + (1.0 - rho) * nb
        return np.where(is_zero, zi[None, :] + (1.0 - zi)[None, :] * count, (1.0 - zi)[None, :] * count)
    if model == "ZIP--ZINB":
        rho = float(par["rho"])
        pi = expit(Z @ par["gamma_p"])
        omega = expit(Z @ par["gamma_nb"])
        pois = poisson_pmf_grid(y_grid, inv_mean(X, par["beta_p"], link))
        nb = nb_pmf_grid(y_grid, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
        zip_part = np.where(is_zero, pi[None, :] + (1.0 - pi)[None, :] * pois, (1.0 - pi)[None, :] * pois)
        zinb_part = np.where(is_zero, omega[None, :] + (1.0 - omega)[None, :] * nb, (1.0 - omega)[None, :] * nb)
        return rho * zip_part + (1.0 - rho) * zinb_part
    raise ValueError(model)


def logpmf_observed(model: str, params: np.ndarray, y: np.ndarray, X: np.ndarray, Z: np.ndarray, link: str) -> np.ndarray:
    p, q = X.shape[1], Z.shape[1]
    par = unpack_model(model, params, p, q)
    y = np.asarray(y, dtype=float)
    is_zero = y == 0.0

    if model == "Poisson":
        return poisson_logpmf(y, inv_mean(X, par["beta_p"], link))
    if model == "NB":
        return nb2_logpmf(y, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
    if model == "ZIP":
        pi = np.clip(expit(Z @ par["gamma_p"]), EPS, 1.0 - EPS)
        pois = poisson_logpmf(y, inv_mean(X, par["beta_p"], link))
        out = np.log1p(-pi) + pois
        out[is_zero] = np.logaddexp(np.log(pi[is_zero]), np.log1p(-pi[is_zero]) + pois[is_zero])
        return out
    if model == "ZINB":
        omega = np.clip(expit(Z @ par["gamma_nb"]), EPS, 1.0 - EPS)
        nb = nb2_logpmf(y, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
        out = np.log1p(-omega) + nb
        out[is_zero] = np.logaddexp(np.log(omega[is_zero]), np.log1p(-omega[is_zero]) + nb[is_zero])
        return out
    if model == "P+NB":
        log_p = poisson_logpmf(y, inv_mean(X, par["beta_p"], link))
        log_n = nb2_logpmf(y, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
        rho = np.clip(float(par["rho"]), EPS, 1.0 - EPS)
        return np.logaddexp(np.log(rho) + log_p, np.log1p(-rho) + log_n)
    if model == "Z+P+NB":
        log_p = poisson_logpmf(y, inv_mean(X, par["beta_p"], link))
        log_n = nb2_logpmf(y, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
        rho = np.clip(float(par["rho"]), EPS, 1.0 - EPS)
        zi = np.clip(expit(Z @ par["gamma"]), EPS, 1.0 - EPS)
        count = np.logaddexp(np.log(rho) + log_p, np.log1p(-rho) + log_n)
        out = np.log1p(-zi) + count
        out[is_zero] = np.logaddexp(np.log(zi[is_zero]), np.log1p(-zi[is_zero]) + count[is_zero])
        return out
    if model == "ZIP--ZINB":
        rho = np.clip(float(par["rho"]), EPS, 1.0 - EPS)
        pi = np.clip(expit(Z @ par["gamma_p"]), EPS, 1.0 - EPS)
        omega = np.clip(expit(Z @ par["gamma_nb"]), EPS, 1.0 - EPS)
        log_p = poisson_logpmf(y, inv_mean(X, par["beta_p"], link))
        log_n = nb2_logpmf(y, inv_mean(X, par["beta_nb"], link), float(par["alpha"]))
        zip_part = np.log1p(-pi) + log_p
        zip_part[is_zero] = np.logaddexp(np.log(pi[is_zero]), np.log1p(-pi[is_zero]) + log_p[is_zero])
        zinb_part = np.log1p(-omega) + log_n
        zinb_part[is_zero] = np.logaddexp(np.log(omega[is_zero]), np.log1p(-omega[is_zero]) + log_n[is_zero])
        return np.logaddexp(np.log(rho) + zip_part, np.log1p(-rho) + zinb_part)
    raise ValueError(model)


def zero_probability(model: str, params: np.ndarray, X: np.ndarray, Z: np.ndarray, link: str) -> np.ndarray:
    return pmf_grid(model, params, X, Z, np.array([0.0]), link)[0]


def simulate_from_model(model: str, params: np.ndarray, X: np.ndarray, Z: np.ndarray, rng: np.random.Generator, link: str) -> np.ndarray:
    p, q = X.shape[1], Z.shape[1]
    par = unpack_model(model, params, p, q)
    n = X.shape[0]

    def draw_pois(beta: np.ndarray) -> np.ndarray:
        return rng.poisson(inv_mean(X, beta, link))

    def draw_nb(beta: np.ndarray, alpha: float) -> np.ndarray:
        mu = inv_mean(X, beta, link)
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
        y_p = draw_pois(par["beta_p"])
        y_n = draw_nb(par["beta_nb"], float(par["alpha"]))
        return np.where(from_p, y_p, y_n).astype(float)
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


def fit_all_models(
    spec: DatasetSpec,
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    link: str,
    *,
    seed: int,
    n_starts: int,
    maxiter: int,
    compute_covariance: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fits: dict[str, object] = {}
    failures: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    p, q = X.shape[1], Z.shape[1]
    done: dict[tuple[str, str], object] = {}
    for i, model in enumerate(MODEL_ORDER):
        t0 = time.time()
        try:
            fit = fit_model(
                model,
                y,
                X,
                Z,
                link=link,
                n_starts=n_starts,
                maxiter=maxiter,
                seed=seed + 1009 * (i + 1),
                extra_starts=warm_starts(model, p, q, done, link),
                compute_covariance=compute_covariance,
            )
            fits[model] = fit
            done[(model, link)] = fit
            row = result_row(spec.key, fit)
            selection_rows.append(row)
            runtime_rows.append(
                {
                    "Dataset": spec.key,
                    "Link": link,
                    "Model": model,
                    "fit_elapsed_sec": time.time() - t0,
                    "success": bool(fit.success),
                    "nit": int(fit.nit),
                    "llf": float(fit.llf),
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                    "grad_norm_inf": float(fit.grad_norm),
                    "cov_status": fit.cov_status,
                }
            )
        except Exception as exc:
            failures.append({"Dataset": spec.key, "Link": link, "Model": model, "Error": repr(exc)})
    return fits, runtime_rows, selection_rows, failures


def diagnostic_rows(spec: DatasetSpec, link: str, y: np.ndarray, X: np.ndarray, Z: np.ndarray, full_params: np.ndarray) -> dict[str, list[dict[str, object]]]:
    max_observed = int(np.nanmax(y)) if len(y) else 0
    max_count = int(min(max(15, max_observed), 80))
    y_grid = np.arange(max_count + 1, dtype=float)
    probs = pmf_grid(FULL_MODEL, full_params, X, Z, y_grid, link)
    expected = probs.sum(axis=1)
    observed = np.bincount(np.asarray(y, dtype=int), minlength=max_count + 1)[: max_count + 1].astype(float)
    tail_observed = float(np.sum(y > max_count))
    tail_expected = float(np.sum(np.clip(1.0 - probs.sum(axis=0), 0.0, 1.0)))

    freq_rows = []
    for yy, obs, exp in zip(y_grid.astype(int), observed, expected):
        freq_rows.append(
            {
                "Dataset": spec.key,
                "Link": link,
                "Count": int(yy),
                "Observed": obs,
                "Expected": exp,
                "RootogramDiff": np.sqrt(obs) - np.sqrt(max(exp, 0.0)),
            }
        )
    freq_rows.append(
        {
            "Dataset": spec.key,
            "Link": link,
            "Count": f">{max_count}",
            "Observed": tail_observed,
            "Expected": tail_expected,
            "RootogramDiff": np.sqrt(tail_observed) - np.sqrt(max(tail_expected, 0.0)),
        }
    )

    dose = X[:, 1]
    unique_dose = np.unique(dose)
    zprob = zero_probability(FULL_MODEL, full_params, X, Z, link)
    zero_rows = []
    if unique_dose.size <= 20:
        for b, dose_value in enumerate(unique_dose, start=1):
            mask = dose == dose_value
            zero_rows.append(
                {
                    "Dataset": spec.key,
                    "Link": link,
                    "Bin": b,
                    "DoseMin": float(dose_value),
                    "DoseMax": float(dose_value),
                    "n": int(np.sum(mask)),
                    "ObservedZeroRate": float(np.mean(y[mask] == 0)),
                    "FittedZeroProbability": float(np.mean(zprob[mask])),
                }
            )
    else:
        bins = np.unique(np.quantile(dose, np.linspace(0.0, 1.0, 11)))
        labels = np.digitize(dose, bins[1:-1], right=True)
        for b in range(bins.size - 1):
            mask = labels == b
            if not np.any(mask):
                continue
            zero_rows.append(
                {
                    "Dataset": spec.key,
                    "Link": link,
                    "Bin": b + 1,
                    "DoseMin": float(np.min(dose[mask])),
                    "DoseMax": float(np.max(dose[mask])),
                    "n": int(np.sum(mask)),
                    "ObservedZeroRate": float(np.mean(y[mask] == 0)),
                    "FittedZeroProbability": float(np.mean(zprob[mask])),
                }
            )

    tail_rows = []
    for threshold in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]:
        if threshold > max_count:
            continue
        expected_tail = float(np.sum(np.clip(1.0 - probs[:threshold, :].sum(axis=0), 0.0, 1.0)))
        observed_tail = float(np.sum(y >= threshold))
        tail_rows.append(
            {
                "Dataset": spec.key,
                "Link": link,
                "Threshold": threshold,
                "ObservedTailCount": observed_tail,
                "ExpectedTailCount": expected_tail,
                "ObservedTailRate": observed_tail / len(y),
                "ExpectedTailRate": expected_tail / len(y),
            }
        )

    max_rqr = int(max(max_count, max_observed))
    y_int = np.asarray(y, dtype=int)
    cdf_grid = np.cumsum(pmf_grid(FULL_MODEL, full_params, X, Z, np.arange(max_rqr + 1, dtype=float), link), axis=0)
    lower = np.zeros(len(y), dtype=float)
    upper = np.ones(len(y), dtype=float)
    within = y_int <= max_rqr
    cols = np.where(within)[0]
    upper[within] = cdf_grid[y_int[within], cols]
    positive = within & (y_int > 0)
    pos_cols = np.where(positive)[0]
    lower[positive] = cdf_grid[y_int[positive] - 1, pos_cols]
    rng = np.random.default_rng(74_211 + sum(ord(c) for c in f"{spec.key}_{link}"))
    u = lower + rng.random(len(y)) * np.maximum(upper - lower, EPS)
    resid = norm.ppf(np.clip(u, 1e-9, 1.0 - 1e-9))
    residual_rows = [
        {
            "Dataset": spec.key,
            "Link": link,
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


def plot_diagnostics(out_dir: Path) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    freq_path = out_dir / "diagnostic_fitted_frequency.csv"
    if freq_path.exists() and freq_path.stat().st_size > 0:
        freq = pd.read_csv(freq_path)
        for (dataset, link), sub in freq.groupby(["Dataset", "Link"], sort=False):
            sub_plot = sub[sub["Count"].astype(str).str.startswith(">") == False].copy()
            counts = sub_plot["Count"].astype(int)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(counts - 0.2, sub_plot["Observed"], width=0.4, label="Observed", alpha=0.75)
            ax.bar(counts + 0.2, sub_plot["Expected"], width=0.4, label="Fitted", alpha=0.75)
            ax.set_xlabel("Count")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{dataset}, {link}: observed and fitted frequencies")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"fitted_frequency_{dataset}_{link}.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.bar(counts, sub_plot["RootogramDiff"], alpha=0.8)
            ax.set_xlabel("Count")
            ax.set_ylabel(r"$\sqrt{O_y}-\sqrt{E_y}$")
            ax.set_title(f"{dataset}, {link}: hanging rootogram residuals")
            fig.tight_layout()
            fig.savefig(figures / f"rootogram_{dataset}_{link}.png", dpi=180)
            plt.close(fig)

    zero_path = out_dir / "diagnostic_zero_probabilities.csv"
    if zero_path.exists() and zero_path.stat().st_size > 0:
        zero = pd.read_csv(zero_path)
        for (dataset, link), sub in zero.groupby(["Dataset", "Link"], sort=False):
            fig, ax = plt.subplots(figsize=(7, 4))
            mids = 0.5 * (sub["DoseMin"] + sub["DoseMax"])
            ax.plot(mids, sub["ObservedZeroRate"], marker="o", label="Observed zero rate")
            ax.plot(mids, sub["FittedZeroProbability"], marker="s", label="Fitted zero probability")
            ax.set_xlabel("Dose")
            ax.set_ylabel("Zero probability")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{dataset}, {link}: zero-probability calibration")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / f"zero_probability_{dataset}_{link}.png", dpi=180)
            plt.close(fig)


def summarise_tables(out_dir: Path) -> None:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    specs = [
        ("runtime", "raw_runtime.csv", ["Dataset", "Link", "Model"], ["fit_elapsed_sec", "grad_norm_inf"]),
        ("cv_log_scores", "raw_cv_log_scores.csv", ["Dataset", "Link", "Model"], ["mean_test_log_score"]),
        ("bootstrap_lrt", "raw_bootstrap_lrt.csv", ["Dataset", "Link", "RestrictedModel"], ["BootstrapPValue", "ObservedLRT", "BootstrapQ95"]),
    ]
    for name, filename, group_cols, value_cols in specs:
        path = out_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        rows = []
        for keys, group in df.groupby(group_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            for col in value_cols:
                vals = pd.to_numeric(group[col], errors="coerce")
                row[f"mean_{col}"] = float(vals.mean())
                row[f"sd_{col}"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
                row[f"n_{col}"] = int(vals.notna().sum())
            rows.append(row)
        pd.DataFrame(rows).to_csv(tables / f"summary_{name}.csv", index=False)
    plot_diagnostics(out_dir)


def run_diagnostics(task: dict[str, object]) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    spec = DATASET_BY_KEY[str(task["dataset"])]
    link = str(task["link"])
    y, X, Z = load_design(spec)
    fits, runtime_rows, selection_rows, failures = fit_all_models(
        spec,
        y,
        X,
        Z,
        link,
        seed=int(task["seed"]),
        n_starts=int(task["n_starts"]),
        maxiter=int(task["maxiter"]),
        compute_covariance=False,
    )
    diag = {"freq": [], "zero": [], "tail": [], "residual": []}
    if FULL_MODEL in fits:
        diag = diagnostic_rows(spec, link, y, X, Z, fits[FULL_MODEL].params)
    return {
        "runtime_rows": runtime_rows,
        "selection_rows": selection_rows,
        "failure_rows": failures,
        "freq_rows": diag["freq"],
        "zero_rows": diag["zero"],
        "tail_rows": diag["tail"],
        "residual_rows": diag["residual"],
    }


def run_cv(task: dict[str, object]) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    spec = DATASET_BY_KEY[str(task["dataset"])]
    link = str(task["link"])
    fold = int(task["fold"])
    cv_folds = int(task["cv_folds"])
    seed = int(task["seed"])
    y, X, Z = load_design(spec)
    rng = np.random.default_rng(seed + 10_003 * fold)
    perm = rng.permutation(len(y))
    folds = np.array_split(perm, cv_folds)
    test_idx = folds[fold]
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=False)

    fits, runtime_rows, _selection_rows, failures = fit_all_models(
        spec,
        y[train_idx],
        X[train_idx, :],
        Z[train_idx, :],
        link,
        seed=seed + 100_003 * fold,
        n_starts=int(task["n_starts"]),
        maxiter=int(task["maxiter"]),
        compute_covariance=False,
    )
    rows = []
    for model, fit in fits.items():
        try:
            ll_vec = logpmf_observed(model, fit.params, y[test_idx], X[test_idx, :], Z[test_idx, :], link)
            rows.append(
                {
                    "Dataset": spec.key,
                    "Link": link,
                    "Fold": fold,
                    "Model": model,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_loglik": float(np.sum(ll_vec)),
                    "mean_test_log_score": float(np.mean(ll_vec)),
                }
            )
        except Exception as exc:
            failures.append({"Dataset": spec.key, "Link": link, "Fold": fold, "Model": f"{model}__cv_logscore", "Error": repr(exc)})
    for row in runtime_rows:
        row["Fold"] = fold
    return {"cv_rows": rows, "runtime_rows": runtime_rows, "failure_rows": failures}


def run_bootstrap(task: dict[str, object]) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    spec = DATASET_BY_KEY[str(task["dataset"])]
    link = str(task["link"])
    restricted_model = str(task["restricted_model"])
    seed = int(task["seed"])
    bootstrap_reps = int(task["bootstrap_reps"])
    y, X, Z = load_design(spec)
    fits, runtime_rows, selection_rows, failures = fit_all_models(
        spec,
        y,
        X,
        Z,
        link,
        seed=seed,
        n_starts=int(task["n_starts"]),
        maxiter=int(task["maxiter"]),
        compute_covariance=False,
    )
    if FULL_MODEL not in fits or restricted_model not in fits:
        return {
            "bootstrap_rows": [],
            "runtime_rows": runtime_rows,
            "selection_rows": selection_rows,
            "failure_rows": failures,
        }

    full = fits[FULL_MODEL]
    restricted = fits[restricted_model]
    p, q = X.shape[1], Z.shape[1]
    observed = max(0.0, 2.0 * (full.llf - restricted.llf))
    rng = np.random.default_rng(seed + 31_415)
    boot_stats: list[float] = []
    t0 = time.time()
    for b in range(bootstrap_reps):
        try:
            yb = simulate_from_model(restricted_model, restricted.params, X, Z, rng, link)
            fit_r = fit_model(
                restricted_model,
                yb,
                X,
                Z,
                link=link,
                n_starts=max(1, min(3, int(task["n_starts"]))),
                maxiter=int(task["maxiter"]),
                seed=int(rng.integers(1, 2**31 - 1)),
                extra_starts=[restricted.params],
                compute_covariance=False,
            )
            fit_f = fit_model(
                FULL_MODEL,
                yb,
                X,
                Z,
                link=link,
                n_starts=max(1, min(3, int(task["n_starts"]))),
                maxiter=int(task["maxiter"]),
                seed=int(rng.integers(1, 2**31 - 1)),
                extra_starts=[full_start_from_restricted(restricted_model, fit_r.params, p, q), full.params],
                compute_covariance=False,
            )
            boot_stats.append(max(0.0, 2.0 * (fit_f.llf - fit_r.llf)))
        except Exception as exc:
            failures.append(
                {
                    "Dataset": spec.key,
                    "Link": link,
                    "RestrictedModel": restricted_model,
                    "BootstrapRep": b,
                    "Error": repr(exc),
                }
            )

    boot = np.asarray(boot_stats, dtype=float)
    p_boot = np.nan if boot.size == 0 else float((1.0 + np.sum(boot >= observed)) / (boot.size + 1.0))
    row = {
        "Dataset": spec.key,
        "Link": link,
        "RestrictedModel": restricted_model,
        "ObservedLRT": observed,
        "BootstrapPValue": p_boot,
        "BootstrapRepsRequested": bootstrap_reps,
        "BootstrapRepsSucceeded": int(boot.size),
        "BootstrapMeanLRT": float(np.mean(boot)) if boot.size else np.nan,
        "BootstrapQ95": float(np.quantile(boot, 0.95)) if boot.size else np.nan,
        "ElapsedSec": time.time() - t0,
    }
    return {
        "bootstrap_rows": [row],
        "runtime_rows": runtime_rows,
        "selection_rows": selection_rows,
        "failure_rows": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extended diagnostics for the real analytical ZIP--ZINB fits.")
    parser.add_argument("--mode", choices=("diagnostics", "cv", "bootstrap", "all"), default="diagnostics")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--links", default="all")
    parser.add_argument("--bootstrap-models", default="all")
    parser.add_argument("--bootstrap-reps", type=int, default=99)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", default="auto")
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "robustness_checks" / "statscomp_real_extended")
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    links = parse_links(args.links)
    models = parse_models(args.bootstrap_models)
    n_jobs = resolve_jobs(str(args.n_jobs))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = {
        "raw_runtime.csv": ["Dataset", "Link", "Fold", "Model", "fit_elapsed_sec", "success", "nit", "llf", "aic", "bic", "grad_norm_inf", "cov_status"],
        "raw_model_selection.csv": [
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
        ],
        "failures.csv": ["Dataset", "Link", "Fold", "Model", "RestrictedModel", "BootstrapRep", "Error"],
        "raw_cv_log_scores.csv": ["Dataset", "Link", "Fold", "Model", "n_train", "n_test", "test_loglik", "mean_test_log_score"],
        "raw_bootstrap_lrt.csv": [
            "Dataset",
            "Link",
            "RestrictedModel",
            "ObservedLRT",
            "BootstrapPValue",
            "BootstrapRepsRequested",
            "BootstrapRepsSucceeded",
            "BootstrapMeanLRT",
            "BootstrapQ95",
            "ElapsedSec",
        ],
        "diagnostic_fitted_frequency.csv": ["Dataset", "Link", "Count", "Observed", "Expected", "RootogramDiff"],
        "diagnostic_zero_probabilities.csv": ["Dataset", "Link", "Bin", "DoseMin", "DoseMax", "n", "ObservedZeroRate", "FittedZeroProbability"],
        "diagnostic_tail.csv": ["Dataset", "Link", "Threshold", "ObservedTailCount", "ExpectedTailCount", "ObservedTailRate", "ExpectedTailRate"],
        "diagnostic_rqr_summary.csv": ["Dataset", "Link", "n", "mean", "sd", "q01", "q05", "q50", "q95", "q99"],
    }

    phases = []
    if args.mode in {"diagnostics", "all"}:
        tasks = [
            {
                "dataset": spec.key,
                "link": link,
                "seed": args.seed + 10_000 * i + 101 * j,
                "n_starts": args.n_starts,
                "maxiter": args.maxiter,
            }
            for i, spec in enumerate(datasets)
            for j, link in enumerate(links)
        ]
        phases.append(("diagnostics", tasks, run_diagnostics))
    if args.mode in {"cv", "all"} and args.cv_folds > 1:
        tasks = [
            {
                "dataset": spec.key,
                "link": link,
                "fold": fold,
                "cv_folds": args.cv_folds,
                "seed": args.seed + 20_000 * i + 1000 * j,
                "n_starts": args.n_starts,
                "maxiter": args.maxiter,
            }
            for i, spec in enumerate(datasets)
            for j, link in enumerate(links)
            for fold in range(args.cv_folds)
        ]
        phases.append(("cv", tasks, run_cv))
    if args.mode in {"bootstrap", "all"} and args.bootstrap_reps > 0:
        tasks = [
            {
                "dataset": spec.key,
                "link": link,
                "restricted_model": model,
                "bootstrap_reps": args.bootstrap_reps,
                "seed": args.seed + 30_000 * i + 2000 * j + 127 * k,
                "n_starts": args.n_starts,
                "maxiter": args.maxiter,
            }
            for i, spec in enumerate(datasets)
            for j, link in enumerate(links)
            for k, model in enumerate(models)
        ]
        phases.append(("bootstrap", tasks, run_bootstrap))

    print(f"Output directory: {out_dir}")
    print(f"Mode: {args.mode}; datasets: {', '.join(s.key for s in datasets)}; links: {', '.join(links)}")
    print(f"Parallel workers: {n_jobs}")

    for phase_name, tasks, func in phases:
        print(f"\nPhase {phase_name}: {len(tasks)} task(s)")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(func, task): task for task in tasks}
            with tqdm(total=len(futures), desc=f"Real {phase_name}", unit="task", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        append_rows(
                            out_dir / "failures.csv",
                            [
                                {
                                    "Dataset": task.get("dataset", ""),
                                    "Link": task.get("link", ""),
                                    "Fold": task.get("fold", ""),
                                    "Model": "__worker__",
                                    "RestrictedModel": task.get("restricted_model", ""),
                                    "Error": repr(exc),
                                }
                            ],
                            cols["failures.csv"],
                        )
                        pbar.update(1)
                        continue

                    append_rows(out_dir / "raw_runtime.csv", result.get("runtime_rows", []), cols["raw_runtime.csv"])
                    append_rows(out_dir / "raw_model_selection.csv", result.get("selection_rows", []), cols["raw_model_selection.csv"])
                    append_rows(out_dir / "failures.csv", result.get("failure_rows", []), cols["failures.csv"])
                    append_rows(out_dir / "raw_cv_log_scores.csv", result.get("cv_rows", []), cols["raw_cv_log_scores.csv"])
                    append_rows(out_dir / "raw_bootstrap_lrt.csv", result.get("bootstrap_rows", []), cols["raw_bootstrap_lrt.csv"])
                    append_rows(out_dir / "diagnostic_fitted_frequency.csv", result.get("freq_rows", []), cols["diagnostic_fitted_frequency.csv"])
                    append_rows(out_dir / "diagnostic_zero_probabilities.csv", result.get("zero_rows", []), cols["diagnostic_zero_probabilities.csv"])
                    append_rows(out_dir / "diagnostic_tail.csv", result.get("tail_rows", []), cols["diagnostic_tail.csv"])
                    append_rows(out_dir / "diagnostic_rqr_summary.csv", result.get("residual_rows", []), cols["diagnostic_rqr_summary.csv"])
                    pbar.update(1)

    summarise_tables(out_dir)
    print(f"\nDone. Summaries/figures are in {out_dir}")


if __name__ == "__main__":
    main()
