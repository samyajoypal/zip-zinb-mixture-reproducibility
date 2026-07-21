# ZIP--ZINB mixture reproducibility repository

This repository contains the code, data and generated numerical outputs used
to reproduce the ZIP--ZINB mixture analyses for:

**Generalized Linear Models with Mixtures of Zero-Inflated Count Distributions**

## Contents

- `src/`: analytical likelihood, score and observed-information implementation.
- `scripts/`: real-data, simulation, diagnostics, bootstrap and table-generation scripts.
- `hpc/`: Slurm batch scripts used for the larger bootstrap/diagnostic runs.
- `data/`: the four real-data CSV files used in the biodosimetry applications:
  `C2_long.csv`, `C3_long.csv`, `dic_1h_exposure_60.csv` and
  `dic_4h_exposure_100.csv`.
- `outputs/`: generated CSV summaries, diagnostic outputs and figures.

## Python/R requirements

Python analyses were run with Python 3 and the packages listed in
`requirements.txt`. The robustness table generator requires R with only base
packages.

## Rebuilding key outputs

Real-data analytical fits:

```bash
PYTHONPATH=src python scripts/run_real_analytical.py
PYTHONPATH=src python scripts/make_real_tables_analytical.py
```

Simulation study:

```bash
bash run_full_simulation.sh
```

Robustness table generation after outputs are available:

```bash
Rscript scripts/make_statscomp_robustness_tables.R
```

The larger bootstrap and diagnostic analyses were run on an HPC cluster using
the Slurm scripts in `hpc/`.
