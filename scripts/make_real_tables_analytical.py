"""Create publication-ready real-data tables from analytical model outputs.

This script is intentionally downstream-only: it reads the CSV files written by
``scripts/run_real_analytical.py`` and writes table fragments under
``outputs/real_analytical/tables``.  It does not import or modify the copied
legacy model code.

Table formatting implemented here:

* model row order: Poisson -> NB -> ZIP -> ZINB -> P+NB -> Z+P+NB -> ZIP--ZINB;
* all log-link rows are italicized in LaTeX;
* alpha and rho are merged into the beta/gamma coefficient tables, so the old
  separate derived-moments tables are no longer regenerated.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "real_analytical"
OUT_DIR = INPUT_DIR / "tables"

MODEL_ORDER = ("Poisson", "NB", "ZIP", "ZINB", "P+NB", "Z+P+NB", "ZIP--ZINB")
LINKS = ("identity", "log")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    caption_label: str
    coef_label: str
    coef_tex_label: str
    beta_terms: tuple[str, ...]
    gamma_terms: tuple[str, ...]


DATASETS = (
    DatasetSpec("C2", "C2", "C2", "C2", ("dose", "dose2"), ("dose",)),
    DatasetSpec("C3", "C3", "C3", "C3", ("dose", "dose2"), ("dose",)),
    DatasetSpec("H2AX1H60", "H2AX1H60", "1H60", r"\texttt{1H60}", ("dose",), ("dose",)),
    DatasetSpec("H2AX4H100", "H2AX4H100", "4H100", r"\texttt{4H100}", ("dose",), ("dose",)),
)


LRT_MODELS = ("Poisson", "NB", "P+NB", "ZIP", "ZINB", "Z+P+NB")
LRT_BOUNDARY_DIMS = {
    "Poisson": 2,
    "NB": 2,
    "P+NB": 2,
    "ZIP": 1,
    "ZINB": 1,
    "Z+P+NB": 0,
}


def _read_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = pd.read_csv(INPUT_DIR / "model_selection_real_analytical.csv")
    params = pd.read_csv(INPUT_DIR / "params_real_analytical.csv")
    return selection, params


def _number_parts(value: float, digits: int = 4) -> tuple[str, bool]:
    """Return (content, is_math) for LaTeX number formatting."""

    if pd.isna(value):
        return "--", False
    value = float(value)
    if value != 0.0 and (abs(value) >= 1e4 or abs(value) < 1e-4):
        mantissa, exponent = f"{value:.3e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}", True
    return f"{value:.{digits}f}", False


def latex_number(value: float, *, italic: bool = False, bold: bool = False, digits: int = 4) -> str:
    content, is_math = _number_parts(value, digits=digits)
    if content == "--":
        return content
    if is_math:
        if bold:
            return rf"$\boldsymbol{{{content}}}$"
        if italic:
            return rf"$\mathit{{{content}}}$"
        return rf"${content}$"
    if bold and italic:
        return rf"\textbf{{\textit{{{content}}}}}"
    if bold:
        return rf"\textbf{{{content}}}"
    if italic:
        return rf"\textit{{{content}}}"
    return content


def latex_link(link: str) -> str:
    return r"\textit{log}" if link == "log" else "identity"


def latex_lrt_link(link: str) -> str:
    return r"\textit{Log}" if link == "log" else "Identity"


def _write_model_selection(selection: pd.DataFrame) -> None:
    all_rows = []
    for spec in DATASETS:
        out = selection[selection.Dataset == spec.key].copy()
        out = out.set_index(["Model", "Link"]).loc[
            pd.MultiIndex.from_product([MODEL_ORDER, LINKS], names=["Model", "Link"])
        ].reset_index()
        out.to_csv(OUT_DIR / f"model_selection_{spec.key}.csv", index=False)
        all_rows.append(out)

        best_by_link: dict[tuple[str, str], float] = {}
        for link in LINKS:
            part = out[out.Link == link]
            best_by_link[(link, "Log-Likelihood")] = part["Log-Likelihood"].max()
            best_by_link[(link, "AIC")] = part["AIC"].min()
            best_by_link[(link, "BIC")] = part["BIC"].min()

        def cell(row: pd.Series, column: str, link: str) -> str:
            target = best_by_link[(link, column)]
            bold = np.isclose(float(row[column]), float(target), rtol=0.0, atol=5e-7)
            return latex_number(row[column], italic=(link == "log"), bold=bold)

        lines = [
            r"\begin{table}[!h]",
            rf"\caption{{Comparison of count regression models on Dataset {spec.caption_label}. "
            r"NB denotes the Negative Binomial model with NB2 parameterization.}",
            r"\centering",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Model & Link & Log-Likelihood & AIC & BIC \\",
            r"\midrule",
        ]
        for model in MODEL_ORDER:
            part = out[out.Model == model].set_index("Link")
            for i, link in enumerate(LINKS):
                row = part.loc[link]
                prefix = rf"\multirow[t]{{2}}{{*}}{{{model}}}" if i == 0 else ""
                lines.append(
                    f"{prefix} & {latex_link(link)} & "
                    f"{cell(row, 'Log-Likelihood', link)} & {cell(row, 'AIC', link)} & "
                    f"{cell(row, 'BIC', link)} \\\\"
                )
            lines.append("")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\label{{tab:{spec.key}}}",
            r"\end{table}",
            "",
        ]
        (OUT_DIR / f"model_selection_{spec.key}.tex").write_text("\n".join(lines), encoding="utf-8")

    pd.concat(all_rows, ignore_index=True).to_csv(OUT_DIR / "model_selection_all.csv", index=False)


def _param_vector(params: pd.DataFrame, dataset: str, model: str, link: str) -> np.ndarray:
    part = params[(params.Dataset == dataset) & (params.Model == model) & (params.Link == link)]
    part = part.sort_values("Index")
    if part.empty:
        raise KeyError(f"Missing parameter vector for {dataset}/{model}/{link}")
    return part["Value"].to_numpy(dtype=float)


def _coef_columns(beta_terms: tuple[str, ...], gamma_terms: tuple[str, ...]) -> list[str]:
    columns = ["Data", "Dataset", "Model", "Link", "rho", "alpha"]
    for prefix, terms in (
        ("pzip_beta", ("0", *beta_terms)),
        ("pzip_gamma", ("0", *gamma_terms)),
        ("nb_beta", ("0", *beta_terms)),
        ("nb_gamma", ("0", *gamma_terms)),
        ("shared_gamma", ("0", *gamma_terms)),
    ):
        columns.extend(f"{prefix}_{term}" for term in terms)
    columns.append("CovarianceStatus")
    return columns


def _empty_coef_row(spec: DatasetSpec, model: str, link: str, cov_status: str) -> dict[str, object]:
    row: dict[str, object] = {
        "Data": spec.coef_label,
        "Dataset": spec.key,
        "Model": model,
        "Link": link,
        "rho": np.nan,
        "alpha": np.nan,
        "CovarianceStatus": cov_status,
    }
    for col in _coef_columns(spec.beta_terms, spec.gamma_terms):
        row.setdefault(col, np.nan)
    return row


def _fill_terms(row: dict[str, object], prefix: str, terms: tuple[str, ...], values: np.ndarray) -> None:
    labels = ("0", *terms)
    for label, value in zip(labels, np.asarray(values, dtype=float)):
        row[f"{prefix}_{label}"] = value


def _coefficient_table(selection: pd.DataFrame, params: pd.DataFrame, specs: tuple[DatasetSpec, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cov_lookup = selection.set_index(["Dataset", "Model", "Link"])["cov_status"].to_dict()
    for spec in specs:
        p = 1 + len(spec.beta_terms)
        q = 1 + len(spec.gamma_terms)
        for model in MODEL_ORDER:
            for link in LINKS:
                theta = _param_vector(params, spec.key, model, link)
                cov_status = str(cov_lookup[(spec.key, model, link)])
                row = _empty_coef_row(spec, model, link, cov_status)

                if model == "Poisson":
                    _fill_terms(row, "pzip_beta", spec.beta_terms, theta[:p])
                elif model == "NB":
                    _fill_terms(row, "nb_beta", spec.beta_terms, theta[:p])
                    row["alpha"] = float(np.exp(theta[-1]))
                elif model == "ZIP":
                    _fill_terms(row, "pzip_beta", spec.beta_terms, theta[:p])
                    _fill_terms(row, "pzip_gamma", spec.gamma_terms, theta[p : p + q])
                elif model == "ZINB":
                    _fill_terms(row, "nb_beta", spec.beta_terms, theta[:p])
                    _fill_terms(row, "nb_gamma", spec.gamma_terms, theta[p : p + q])
                    row["alpha"] = float(np.exp(theta[-1]))
                elif model == "P+NB":
                    row["rho"] = float(expit(theta[0]))
                    _fill_terms(row, "pzip_beta", spec.beta_terms, theta[1 : 1 + p])
                    _fill_terms(row, "nb_beta", spec.beta_terms, theta[1 + p : 1 + 2 * p])
                    row["alpha"] = float(np.exp(theta[-1]))
                elif model == "Z+P+NB":
                    row["rho"] = float(expit(theta[0]))
                    _fill_terms(row, "pzip_beta", spec.beta_terms, theta[1 : 1 + p])
                    _fill_terms(row, "nb_beta", spec.beta_terms, theta[1 + p : 1 + 2 * p])
                    _fill_terms(row, "shared_gamma", spec.gamma_terms, theta[1 + 2 * p : 1 + 2 * p + q])
                    row["alpha"] = float(np.exp(theta[-1]))
                elif model == "ZIP--ZINB":
                    row["rho"] = float(expit(theta[0]))
                    _fill_terms(row, "pzip_beta", spec.beta_terms, theta[1 : 1 + p])
                    _fill_terms(row, "pzip_gamma", spec.gamma_terms, theta[1 + p : 1 + p + q])
                    nb_start = 1 + p + q
                    _fill_terms(row, "nb_beta", spec.beta_terms, theta[nb_start : nb_start + p])
                    _fill_terms(row, "nb_gamma", spec.gamma_terms, theta[nb_start + p : nb_start + p + q])
                    row["alpha"] = float(np.exp(theta[-1]))
                else:  # pragma: no cover - guarded by MODEL_ORDER
                    raise ValueError(model)
                rows.append(row)

    table = pd.DataFrame(rows)
    return table.loc[:, _coef_columns(specs[0].beta_terms, specs[0].gamma_terms)]


def _latex_term(symbol: str, term: str) -> str:
    if term == "0":
        sub = "0"
    elif term == "dose":
        sub = r"\mathrm{dose}"
    elif term == "dose2":
        sub = r"\mathrm{dose}^2"
    else:
        sub = rf"\mathrm{{{term}}}"
    return rf"${symbol}_{{{sub}}}$"


def _write_coefficient_tex(table: pd.DataFrame, specs: tuple[DatasetSpec, ...], stem: str, caption: str, label: str) -> None:
    beta_terms = ("0", *specs[0].beta_terms)
    gamma_terms = ("0", *specs[0].gamma_terms)
    n_beta = len(beta_terms)
    n_gamma = len(gamma_terms)
    pzip_cols = n_beta + n_gamma
    nb_cols = n_beta + n_gamma
    shared_cols = n_gamma
    total_cols = 3 + 2 + pzip_cols + nb_cols + shared_cols

    alignment = "l l l " + " ".join("c" for _ in range(total_cols - 3))
    scalar_end = 5
    pzip_start = 6
    pzip_end = pzip_start + pzip_cols - 1
    nb_start = pzip_end + 1
    nb_end = nb_start + nb_cols - 1
    shared_start = nb_end + 1
    shared_end = shared_start + shared_cols - 1

    header_beta = " & ".join(_latex_term(r"\beta", term) for term in beta_terms)
    header_gamma = " & ".join(_latex_term(r"\gamma", term) for term in gamma_terms)

    lines = [
        r"\begin{sidewaystable}[!p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        "",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        "",
        r"\resizebox{0.98\textheight}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        "Data & Model & Link",
        r"& \multicolumn{2}{c}{Scalars}",
        rf"& \multicolumn{{{pzip_cols}}}{{c}}{{Poisson/ZIP component}}",
        rf"& \multicolumn{{{nb_cols}}}{{c}}{{NB/ZINB component}}",
        rf"& \multicolumn{{{shared_cols}}}{{c}}{{Shared ZI}} \\",
        rf"\cmidrule(lr){{4-{scalar_end}}}\cmidrule(lr){{{pzip_start}-{pzip_end}}}"
        rf"\cmidrule(lr){{{nb_start}-{nb_end}}}\cmidrule(lr){{{shared_start}-{shared_end}}}",
        "& & & \\multicolumn{2}{c}{} "
        + rf"& \multicolumn{{{n_beta}}}{{c}}{{$\beta$}} & \multicolumn{{{n_gamma}}}{{c}}{{$\gamma$}} "
        + rf"& \multicolumn{{{n_beta}}}{{c}}{{$\beta$}} & \multicolumn{{{n_gamma}}}{{c}}{{$\gamma$}} "
        + rf"& \multicolumn{{{n_gamma}}}{{c}}{{$\gamma$}} \\",
        "& & & $\\rho$ & $\\alpha$ "
        + f"& {header_beta} & {header_gamma} "
        + f"& {header_beta} & {header_gamma} "
        + f"& {header_gamma} \\\\",
        r"\midrule",
        "",
    ]

    value_cols = (
        ["rho", "alpha"]
        + [f"pzip_beta_{t}" for t in beta_terms]
        + [f"pzip_gamma_{t}" for t in gamma_terms]
        + [f"nb_beta_{t}" for t in beta_terms]
        + [f"nb_gamma_{t}" for t in gamma_terms]
        + [f"shared_gamma_{t}" for t in gamma_terms]
    )
    for sidx, spec in enumerate(specs):
        part = table[table.Dataset == spec.key].copy()
        part = part.set_index(["Model", "Link"]).loc[
            pd.MultiIndex.from_product([MODEL_ORDER, LINKS], names=["Model", "Link"])
        ].reset_index()
        for ridx, row in part.iterrows():
            data = rf"\multirow{{{len(part)}}}{{*}}{{{spec.coef_tex_label}}}" if ridx == 0 else ""
            italic = row["Link"] == "log"
            values = " & ".join(latex_number(row[c], italic=italic) for c in value_cols)
            lines.append(f"{data} & {row.Model} & {latex_link(row.Link)} & {values} \\\\")
            if ridx % 2 == 1 and ridx != len(part) - 1:
                lines.append("")
        if sidx != len(specs) - 1:
            lines += ["", r"\midrule", ""]

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}%",
        r"\end{sidewaystable}",
        "",
    ]
    (OUT_DIR / f"{stem}.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_coefficient_tables(selection: pd.DataFrame, params: pd.DataFrame) -> None:
    c_specs = DATASETS[:2]
    h_specs = DATASETS[2:]
    c_table = _coefficient_table(selection, params, c_specs)
    h_table = _coefficient_table(selection, params, h_specs)
    c_table.to_csv(OUT_DIR / "beta_gamma_scalar_C2_C3.csv", index=False)
    h_table.to_csv(OUT_DIR / "beta_gamma_scalar_H2AX.csv", index=False)

    _write_coefficient_tex(
        c_table,
        c_specs,
        "beta_gamma_scalar_C2_C3",
        "Count-regression beta, zero-inflation gamma, and scalar parameter estimates for datasets C2 and C3. "
        "Beta includes dose-squared; gamma is linear in dose. The scalar columns report natural-scale "
        r"$\rho$ and $\alpha$. NB denotes the NB2 parameterization.",
        "tab:c2c3_model_betas",
    )
    _write_coefficient_tex(
        h_table,
        h_specs,
        "beta_gamma_scalar_H2AX",
        r"Count-regression beta, zero-inflation gamma, and scalar parameter estimates for the $\gamma$-H2AX datasets. "
        r"The scalar columns report natural-scale $\rho$ and $\alpha$. NB denotes the NB2 parameterization.",
        "tab:H2AX_model_betas",
    )


def _chibar_pvalue(stat: float, k: int) -> tuple[float, str]:
    if k <= 0:
        raise ValueError("k must be positive for chi-bar calculation")
    dfs = np.arange(k + 1, dtype=int)
    weights = np.array([comb(k, int(df)) for df in dfs], dtype=float) / (2.0**k)
    tails = chi2.sf(max(0.0, stat), dfs).astype(float)
    tails[dfs == 0] = 1.0 if stat <= 0.0 else 0.0
    pval = float(np.dot(weights, tails))
    return pval, f"chi-bar² heuristic; dfs={list(map(int, dfs))}; weights={weights.round(6).tolist()}"


def _lrt_tables(selection: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = selection.set_index(["Dataset", "Model", "Link"])
    detailed_rows: list[dict[str, object]] = []
    for spec in DATASETS:
        for link in LINKS:
            full = lookup.loc[(spec.key, "ZIP--ZINB", link)]
            ll_full = float(full["Log-Likelihood"])
            k_full = int(full["n_params"])
            for model in LRT_MODELS:
                restricted = lookup.loc[(spec.key, model, link)]
                ll_r = float(restricted["Log-Likelihood"])
                k_r = int(restricted["n_params"])
                stat = max(0.0, 2.0 * (ll_full - ll_r))
                boundary_dim = LRT_BOUNDARY_DIMS[model]
                df_diff = max(1, k_full - k_r)
                if boundary_dim > 0:
                    pval, method = _chibar_pvalue(stat, boundary_dim)
                    df_report: float | int = np.nan
                else:
                    pval = float(chi2.sf(stat, df_diff))
                    if model == "Z+P+NB":
                        method = f"chi²(df={df_diff}; shared inflation gamma_z=gamma_nb)"
                    else:
                        method = f"chi²(df={df_diff})"
                    df_report = df_diff
                detailed_rows.append(
                    {
                        "Dataset": spec.key,
                        "Link": link,
                        "RestrictedModel": model,
                        "FullModel": "ZIP--ZINB",
                        "RestrictedLogLik": ll_r,
                        "FullLogLik": ll_full,
                        "LRT": stat,
                        "df": df_report,
                        "boundary_dim": boundary_dim,
                        "p_value": pval,
                        "decision": "Reject" if pval < alpha else "Fail",
                        "method": method,
                    }
                )
    detailed = pd.DataFrame(detailed_rows)
    decisions = (
        detailed.pivot(index=["Dataset", "Link"], columns="RestrictedModel", values="decision")
        .reset_index()
        .loc[:, ["Dataset", "Link", *LRT_MODELS]]
    )
    return detailed, decisions


def _write_lrt(selection: pd.DataFrame) -> None:
    detailed, decisions = _lrt_tables(selection)
    detailed.to_csv(OUT_DIR / "lrt_results_detailed.csv", index=False)
    decisions.to_csv(OUT_DIR / "lrt_decisions_all.csv", index=False)

    lines = [
        r"\begin{table}[!h]",
        r"\caption{Likelihood ratio test decisions for real cytogenetic datasets}",
        r"\centering",
        r"\adjustbox{max width=\textwidth}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Dataset & Link & Poisson & NB & P+NB & ZIP & ZINB & Z+P+NB \\",
        r"\midrule",
        "",
    ]
    for didx, spec in enumerate(DATASETS):
        part = decisions[decisions.Dataset == spec.key].set_index("Link")
        for i, link in enumerate(LINKS):
            data = rf"\multirow{{2}}{{*}}{{{spec.key}}}" if i == 0 else ""
            vals = " & ".join(str(part.loc[link, c]) for c in LRT_MODELS)
            lines.append(f"{data} & {latex_lrt_link(link)} & {vals} \\\\")
        if didx != len(DATASETS) - 1:
            lines += ["", r"\midrule", ""]
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\label{tab:lrt_real}",
        r"\end{table}",
        "",
    ]
    (OUT_DIR / "lrt_decisions_all.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_manifest() -> None:
    files = [
        "model_selection_C2.tex",
        "model_selection_C3.tex",
        "model_selection_H2AX1H60.tex",
        "model_selection_H2AX4H100.tex",
        "beta_gamma_scalar_C2_C3.tex",
        "beta_gamma_scalar_H2AX.tex",
        "lrt_decisions_all.tex",
    ]
    text = [
        "Analytical real-data table fragments",
        "====================================",
        "",
        "Generated from outputs/real_analytical/model_selection_real_analytical.csv",
        "and outputs/real_analytical/params_real_analytical.csv.",
        "",
        "Use these as real-data table fragments.",
        "The old derived-moments/parameter tables are intentionally not regenerated;",
        "rho and alpha are merged into the beta/gamma tables as requested.",
        "",
        "LaTeX fragments:",
        *[f"- {name}" for name in files],
        "",
        "CSV companions:",
        "- model_selection_all.csv and model_selection_<dataset>.csv",
        "- beta_gamma_scalar_C2_C3.csv",
        "- beta_gamma_scalar_H2AX.csv",
        "- lrt_results_detailed.csv",
        "- lrt_decisions_all.csv",
        "",
    ]
    (OUT_DIR / "README_tables.txt").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection, params = _read_outputs()
    _write_model_selection(selection)
    _write_coefficient_tables(selection, params)
    _write_lrt(selection)
    _write_manifest()
    print(f"Saved analytical real-data tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
