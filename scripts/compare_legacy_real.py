"""Compare new analytical real-data fits with the old publication-table CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-csv",
        type=Path,
        default=ROOT.parent / "model_selection_all.csv",
        help="Old model-selection CSV to compare against.",
    )
    args = parser.parse_args()

    new_path = ROOT / "outputs" / "real_analytical" / "model_selection_real_analytical.csv"
    out_dir = ROOT / "outputs" / "real_analytical"
    out_dir.mkdir(parents=True, exist_ok=True)

    new = pd.read_csv(new_path)
    old = pd.read_csv(args.legacy_csv)

    merged = new.merge(
        old,
        on=["Dataset", "Model", "Link"],
        how="left",
        suffixes=("_analytical", "_legacy"),
    )
    merged["Delta_LogLik"] = (
        merged["Log-Likelihood_analytical"] - merged["Log-Likelihood_legacy"]
    )
    merged["Delta_AIC"] = merged["AIC_analytical"] - merged["AIC_legacy"]
    merged["Delta_BIC"] = merged["BIC_analytical"] - merged["BIC_legacy"]

    keep = [
        "Dataset",
        "Model",
        "Link",
        "Log-Likelihood_analytical",
        "Log-Likelihood_legacy",
        "Delta_LogLik",
        "AIC_analytical",
        "AIC_legacy",
        "Delta_AIC",
        "BIC_analytical",
        "BIC_legacy",
        "Delta_BIC",
        "success",
        "grad_norm_inf",
    ]
    merged.loc[:, keep].to_csv(out_dir / "legacy_vs_analytical_real.csv", index=False)

    biggest = (
        merged.loc[:, keep]
        .assign(abs_delta=lambda d: d["Delta_LogLik"].abs())
        .sort_values("abs_delta", ascending=False)
        .drop(columns="abs_delta")
        .head(20)
    )
    biggest.to_csv(out_dir / "legacy_vs_analytical_top_differences.csv", index=False)

    print(out_dir / "legacy_vs_analytical_real.csv")
    print(out_dir / "legacy_vs_analytical_top_differences.csv")


if __name__ == "__main__":
    main()
