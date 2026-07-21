"""Combine split Statistics & Computing robustness outputs.

The HPC bootstrap jobs are intentionally split into many independent output
folders to avoid concurrent writes to the same CSV.  Run this after fetching
the HPC outputs to create combined raw files and simple summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def combine_tree(base: Path, pattern: str, combined_dir: Path) -> None:
    combined_dir.mkdir(parents=True, exist_ok=True)
    files_by_name: dict[str, list[Path]] = {}
    for path in sorted(base.glob(pattern)):
        if path.is_file() and path.name.endswith(".csv"):
            files_by_name.setdefault(path.name, []).append(path)
    for name, files in sorted(files_by_name.items()):
        frames = []
        for path in files:
            if path.stat().st_size == 0:
                continue
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            df.insert(0, "SourceFolder", path.parent.name)
            frames.append(df)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(combined_dir / name, index=False)


def summarise_bootstrap(raw_path: Path, out_path: Path, group_cols: list[str]) -> None:
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return
    df = pd.read_csv(raw_path)
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in ("BootstrapPValue", "ObservedLRT", "BootstrapQ95"):
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"mean_{col}"] = float(vals.mean())
            row[f"sd_{col}"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
            row[f"n_{col}"] = int(vals.notna().sum())
        if "BootstrapRepsSucceeded" in group:
            row["min_BootstrapRepsSucceeded"] = int(pd.to_numeric(group["BootstrapRepsSucceeded"], errors="coerce").min())
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root

    synthetic = root / "outputs" / "robustness_checks" / "statscomp_synthetic_bootstrap_all"
    synthetic_combined = synthetic / "combined"
    combine_tree(synthetic, "*/*.csv", synthetic_combined)
    summarise_bootstrap(
        synthetic_combined / "raw_bootstrap_lrt.csv",
        synthetic_combined / "tables" / "summary_bootstrap_lrt.csv",
        ["Scenario", "Dataset", "RestrictedModel"],
    )

    real = root / "outputs" / "robustness_checks" / "statscomp_real_bootstrap_all"
    real_combined = real / "combined"
    combine_tree(real, "*/*.csv", real_combined)
    summarise_bootstrap(
        real_combined / "raw_bootstrap_lrt.csv",
        real_combined / "tables" / "summary_bootstrap_lrt.csv",
        ["Dataset", "Link", "RestrictedModel"],
    )

    print("Combined robustness outputs:")
    print(f"  {synthetic_combined}")
    print(f"  {real_combined}")


if __name__ == "__main__":
    main()
