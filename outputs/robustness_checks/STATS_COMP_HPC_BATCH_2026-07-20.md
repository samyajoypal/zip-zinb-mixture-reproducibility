# Statistics & Computing robustness batch submitted 2026-07-20

Remote project root:

```text
/home/web15red/zip_zinb_analytical_project
```

Submitted jobs:

| Job ID | Script | Purpose |
|---:|---|---|
| 22827193 | `hpc/run_statscomp_synthetic_bootstrap_array.sbatch` | Synthetic parametric-bootstrap LRT, 24 array tasks: D1--D4 × six restrictions, 20 reps, B=99 |
| 22827194 | `hpc/run_statscomp_synthetic_diagnostics.sbatch` | Synthetic fitted-frequency/rootogram, zero-probability, tail, RQR, and log-score diagnostics |
| 22827195 | `hpc/run_statscomp_real_diagnostics.sbatch` | Real-data fitted-frequency/rootogram, zero-probability, tail, RQR, runtime/model-selection diagnostics |
| 22827196 | `hpc/run_statscomp_real_cv.sbatch` | Real-data 5-fold out-of-sample log scores |
| 22827198 | `hpc/run_statscomp_real_bootstrap_array.sbatch` | Real-data parametric-bootstrap LRT, 48 array tasks: four datasets × two links × six restrictions, B=99 |

Partition adjustments:

| Job ID | Status | Note |
|---:|---|---|
| 22827195 | cancelled before running | Replaced by `22827229` on `bigmem`; completed successfully |
| 22827196 | cancelled before running | Replaced by `22827230` on `bigmem`; completed successfully |
| 22827251 | completed successfully | Replacement for real-bootstrap array elements 24--47 on `skylake-96` |
| 22827193 | running | Synthetic bootstrap array throttle increased from 6 to 8 |
| 22827300 | running | Replacement for synthetic-bootstrap array elements 7--8 on `bigmem` |
| 22827301 | running/pending for resources | Replacement for synthetic-bootstrap array elements 9--23 on `cpuidle`; original `22827193_[7-23]` cancelled before running |
| 22827339 | running | Replacement for synthetic-bootstrap array elements 15--17 on `skylake-96` with `--mem=60G`; original pending copies cancelled before running |
| 22827340 | running | Replacement for synthetic-bootstrap array elements 18--19 on `bigmem` with `--mem=60G`; original pending copy for 20 cancelled before running |
| 22827341 | running | Replacement for synthetic-bootstrap array elements 21--22 on `cpuidle` with `--mem=60G`; original pending copy for 23 cancelled before running |
| 22827355 | running | Replacement for synthetic-bootstrap array elements 20 and 23 on `skylake-96` with `--mem=60G` |
| 22830310 | running | Focused B=499 real bootstrap for H2AX4H100 identity P+NB |
| 22830311 | running | Focused B=499 real bootstrap for H2AX4H100 identity ZINB |
| 22830312 | running | Focused B=499 real bootstrap for H2AX4H100 identity Z+P+NB |
| 22830313 | running | Focused B=499 real bootstrap for H2AX4H100 log P+NB |
| 22830314 | running | Focused B=499 real bootstrap for H2AX4H100 log Z+P+NB |

Older completed job fetched:

| Job ID | Script | Purpose |
|---:|---|---|
| 22826817 | `hpc/run_d3_positive_bootstrap_v2.sbatch` | Broad D3 bootstrap check, 12 reps, B=39 |

Monitor:

```bash
ssh web15red@elwe1.rz.rptu.de 'sacct -j 22827193,22827194,22827198,22827229,22827230,22827251,22827300,22827301,22827339,22827340,22827341,22827355,22830310,22830311,22830312,22830313,22830314 --format=JobID,JobName,Partition,State,Elapsed,ExitCode%12,AllocCPUS,MaxRSS -P'
```

Fetch after completion with your preferred `rsync` or `scp` command into `outputs/robustness_checks/`.
