# Statistics & Computing robustness results brief

Date: 2026-07-21

## Completion and output integrity

- All main synthetic bootstrap, real bootstrap, real diagnostic, real cross-validation, and synthetic diagnostic jobs completed successfully.
- The cancelled Slurm entries were deliberate replacements of pending array elements; the replacement jobs completed.
- No `failures.csv` files were produced in the fetched robustness outputs.
- Synthetic bootstrap LRT completeness: 480 rows = 4 scenarios × 6 restrictions × 20 simulation replicates; all rows used 99 successful bootstrap samples.
- Real bootstrap LRT completeness: 48 rows = 4 datasets × 2 links × 6 restrictions; all original rows used 99 successful bootstrap samples.
- Focused high-resolution bootstrap checks were added for borderline H2AX4H100 comparisons; these used 499 bootstrap samples.

## Main synthetic findings

- The parametric-bootstrap LRT rejects every restricted model in every synthetic scenario.
- All synthetic bootstrap p-values are 0.010 with B=99, the smallest attainable value under that design.
- Synthetic diagnostics are strong:
  - randomized quantile residual means range from -0.0066 to 0.0037;
  - randomized quantile residual SDs range from 0.9979 to 1.0069;
  - zero-probability MAE by scenario is below 0.009;
  - maximum tail-rate error by scenario is below 0.0022.
- Out-of-sample synthetic log scores select ZIP--ZINB for D1--D4.

## Main real-data findings

- Bootstrap LRTs strongly reject Poisson and NB for every real dataset/link.
- C2 and C3:
  - Bootstrap LRT often fails to reject zero-inflated or mixture restrictions against ZIP--ZINB.
  - This supports presenting ZIP--ZINB as a flexible benchmark while acknowledging that parsimonious nested models can be adequate for these datasets.
- H2AX1H60:
  - All restrictions are rejected for both identity and log links.
  - This is the strongest real-data support for the full ZIP--ZINB specification.
- H2AX4H100:
  - Identity link: Poisson, NB, P+NB, and Z+P+NB are rejected; ZIP and ZINB are not rejected.
  - Log link: Poisson and NB are rejected; P+NB, ZIP, ZINB, and Z+P+NB are not rejected.
  - Focused B=499 checks refine the borderline values:
    - identity P+NB: p = 0.036, reject;
    - identity ZINB: p = 0.092, fail;
    - identity Z+P+NB: p = 0.016, reject;
    - log P+NB: p = 0.080, fail;
    - log Z+P+NB: p = 0.052, fail but very close to 5%.

## Cross-validated real-data log scores

- ZIP--ZINB is best for H2AX1H60 under both links and for H2AX4H100 with the log link.
- Simpler models have slightly better CV scores in several cases:
  - C2: ZINB is best, but ZIP--ZINB is extremely close.
  - C3: P+NB is best; ZIP--ZINB is close.
  - H2AX4H100 identity: P+NB is best; ZIP--ZINB is close.
- The predictive differences are tiny (best minus ZIP--ZINB is at most about 0.00135 per observation), so the text should not claim that ZIP--ZINB always wins predictively.

## Real-data diagnostics

- C2 and C3 diagnostics look very good:
  - RQR mean close to zero and SD close to one;
  - zero-probability MAE below 0.006;
  - max tail-rate error below 0.002.
- H2AX diagnostics are acceptable but visibly less perfect:
  - zero-probability MAE around 0.026--0.033;
  - max tail-rate error around 0.029--0.042;
  - H2AX4H100 RQR SD is about 0.898 under both links, suggesting slightly compressed residual variability.

## Manuscript recommendation

- Keep the chi-bar/mixed chi-square LRT as the main theoretical test framework.
- Add parametric-bootstrap LRT as a robustness/sensitivity analysis, preferably in an additional Statistics & Computing robustness section or supplementary material.
- Phrase the real-data conclusion carefully:
  - the full model is a useful flexible benchmark and is clearly necessary in H2AX1H60;
  - in C2/C3 and parts of H2AX4H100, simpler nested models often achieve nearly identical predictive performance and sometimes are not rejected by bootstrap LRT.
- Include the CV and diagnostic summaries to strengthen the submission, but avoid overclaiming that the full model dominates all criteria.

## Generated artifacts

- CSV tables: `outputs/robustness_checks/statscomp_tables/`
- LaTeX fragments: `outputs/robustness_checks/statscomp_tables/latex_fragments/`
- Robustness figures: `outputs/robustness_checks/statscomp_tables/figures/robustness/`
- Combined insert fragment: `outputs/robustness_checks/statscomp_tables/latex_fragments/robustness_insert.tex`
