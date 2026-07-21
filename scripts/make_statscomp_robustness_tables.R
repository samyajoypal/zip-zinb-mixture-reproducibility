#!/usr/bin/env Rscript

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)[1]
file_arg <- sub("^--file=", "", file_arg)
root <- normalizePath(file.path(dirname(file_arg), ".."), mustWork = FALSE)
if (is.na(root) || root == "." || !nzchar(root)) root <- "zip_zinb_analytical_project"
if (!dir.exists(root)) root <- "."

rob <- file.path(root, "outputs", "robustness_checks")
out_dir <- file.path(rob, "statscomp_tables")
tex_dir <- file.path(out_dir, "latex_fragments")
fig_dir <- file.path(out_dir, "figures", "robustness")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tex_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

model_order <- c("Poisson", "NB", "ZIP", "ZINB", "P+NB", "Z+P+NB")

read_csv <- function(...) read.csv(file.path(...), stringsAsFactors = FALSE)

fmt <- function(x, digits = 3) {
  ifelse(is.na(x), "--", formatC(x, format = "f", digits = digits))
}

fmt_p <- function(p) {
  ifelse(is.na(p), "--", formatC(p, format = "f", digits = 3))
}

latex_escape <- function(x) {
  x <- gsub("--", "\\mbox{--}", x, fixed = TRUE)
  x <- gsub("&", "\\&", x, fixed = TRUE)
  x
}

latex_dataset <- function(x) {
  ifelse(x == "H2AX1H60", "\\texttt{1H60}",
         ifelse(x == "H2AX4H100", "\\texttt{4H100}", x))
}

latex_link <- function(x) ifelse(x == "log", "\\textit{log}", "identity")

write_simple_tex <- function(path, caption, label, headers, rows, align = NULL, size = "\\small", resize = FALSE) {
  if (is.null(align)) align <- paste(rep("l", length(headers)), collapse = "")
  lines <- c(
    "\\begin{table}[!htbp]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    size,
    "\\setlength{\\tabcolsep}{3pt}",
    "\\renewcommand{\\arraystretch}{1.10}"
  )
  if (resize) lines <- c(lines, "\\begin{adjustbox}{max width=\\textwidth}")
  lines <- c(lines, paste0("\\begin{tabular}{", align, "}"), "\\toprule")
  lines <- c(lines, paste(headers, collapse = " & "), "\\\\", "\\midrule")
  for (i in seq_len(nrow(rows))) {
    lines <- c(lines, paste(as.character(rows[i, ]), collapse = " & "), "\\\\")
  }
  lines <- c(lines, "\\bottomrule", "\\end{tabular}")
  if (resize) lines <- c(lines, "\\end{adjustbox}")
  lines <- c(lines, "\\end{table}", "")
  writeLines(lines, path)
}

syn_boot <- read_csv(rob, "statscomp_synthetic_bootstrap_all", "combined", "raw_bootstrap_lrt.csv")
real_boot <- read_csv(rob, "statscomp_real_bootstrap_all", "combined", "raw_bootstrap_lrt.csv")
real_boot$BootstrapSource <- "B=99"
borderline_files <- Sys.glob(file.path(rob, "statscomp_real_bootstrap_borderline_B499", "*", "raw_bootstrap_lrt.csv"))
if (length(borderline_files) > 0) {
  borderline <- do.call(rbind, lapply(borderline_files, read.csv, stringsAsFactors = FALSE))
  borderline$BootstrapSource <- "B=499"
  write.csv(borderline, file.path(out_dir, "real_bootstrap_borderline_B499_sensitivity.csv"), row.names = FALSE)
}
real_cv <- read_csv(rob, "statscomp_real_cv", "tables", "summary_cv_log_scores.csv")
real_rqr <- read_csv(rob, "statscomp_real_diagnostics", "diagnostic_rqr_summary.csv")
real_zero <- read_csv(rob, "statscomp_real_diagnostics", "diagnostic_zero_probabilities.csv")
real_tail <- read_csv(rob, "statscomp_real_diagnostics", "diagnostic_tail.csv")
syn_log <- read_csv(rob, "statscomp_synthetic_diagnostics", "tables", "summary_test_log_scores.csv")
syn_rqr <- read_csv(rob, "statscomp_synthetic_diagnostics", "diagnostic_rqr_summary.csv")
syn_zero <- read_csv(rob, "statscomp_synthetic_diagnostics", "diagnostic_zero_probabilities.csv")
syn_tail <- read_csv(rob, "statscomp_synthetic_diagnostics", "diagnostic_tail.csv")

# Synthetic bootstrap LRT summary.
syn_boot_rows <- do.call(rbind, lapply(split(syn_boot, paste(syn_boot$Scenario, syn_boot$RestrictedModel, sep = "|")), function(d) {
  data.frame(
    Scenario = d$Scenario[1],
    RestrictedModel = d$RestrictedModel[1],
    RejectionRate = mean(d$BootstrapPValue < 0.05),
    MeanObservedLRT = mean(d$ObservedLRT),
    MeanBootstrapQ95 = mean(d$BootstrapQ95),
    MeanBootstrapP = mean(d$BootstrapPValue),
    MinBootstrapP = min(d$BootstrapPValue),
    MaxBootstrapP = max(d$BootstrapPValue),
    Reps = nrow(d),
    B = min(d$BootstrapRepsSucceeded)
  )
}))
syn_boot_rows <- syn_boot_rows[order(syn_boot_rows$Scenario, match(syn_boot_rows$RestrictedModel, model_order)), ]
write.csv(syn_boot_rows, file.path(out_dir, "synthetic_bootstrap_lrt_summary.csv"), row.names = FALSE)
syn_tex <- data.frame(
  Scenario = syn_boot_rows$Scenario,
  Model = latex_escape(syn_boot_rows$RestrictedModel),
  `Reject rate` = fmt(syn_boot_rows$RejectionRate, 2),
  `Mean LRT` = fmt(syn_boot_rows$MeanObservedLRT, 2),
  `Mean boot. 95\\%` = fmt(syn_boot_rows$MeanBootstrapQ95, 2),
  `Mean p` = fmt_p(syn_boot_rows$MeanBootstrapP),
  check.names = FALSE
)
write_simple_tex(
  file.path(tex_dir, "synthetic_bootstrap_lrt_summary.tex"),
  "Parametric-bootstrap LRT summary for the synthetic datasets. Each entry is based on 20 simulation replicates with 99 bootstrap samples per replicate; p-values of 0.01 are the smallest attainable values under this design.",
  "tab:statscomp_synthetic_bootstrap_lrt",
  names(syn_tex),
  syn_tex,
  align = "llrrrr",
  size = "\\scriptsize"
)

# Asymptotic/chi-bar versus parametric-bootstrap LRT summaries for the
# simulation study.
sim_lrt_file <- file.path(root, "outputs", "sim_analytical", "statscomp_full_extended", "raw_lrt.csv")
if (file.exists(sim_lrt_file)) {
  sim_asym <- read.csv(sim_lrt_file, stringsAsFactors = FALSE)
  sim_asym$AsymReject <- as.character(sim_asym$reject) %in% c("TRUE", "True", "true", "1")
  asym_rows <- do.call(rbind, lapply(split(sim_asym, paste(sim_asym$Scenario, sim_asym$RestrictedModel, sep = "|")), function(d) {
    data.frame(
      Scenario = d$Scenario[1],
      Dataset = d$Dataset[1],
      RestrictedModel = d$RestrictedModel[1],
      AsymRejectRate = mean(d$AsymReject),
      MeanAsymP = mean(d$p_value, na.rm = TRUE),
      RepsAsym = nrow(d),
      Reference = ifelse(any(!is.na(d$boundary_dim) & d$boundary_dim > 0), "chi-bar", "chi-square")
    )
  }))
  boot_rows <- do.call(rbind, lapply(split(syn_boot, paste(syn_boot$Scenario, syn_boot$RestrictedModel, sep = "|")), function(d) {
    data.frame(
      Scenario = d$Scenario[1],
      Dataset = d$Dataset[1],
      RestrictedModel = d$RestrictedModel[1],
      BootRejectRate = mean(d$BootstrapPValue < 0.05),
      MeanBootP = mean(d$BootstrapPValue),
      RepsBoot = nrow(d),
      B = min(d$BootstrapRepsSucceeded)
    )
  }))
  lrt_comp <- merge(asym_rows, boot_rows, by = c("Scenario", "Dataset", "RestrictedModel"), all = TRUE)
  lrt_comp <- lrt_comp[order(lrt_comp$Scenario, match(lrt_comp$RestrictedModel, model_order)), ]
  write.csv(lrt_comp, file.path(out_dir, "simulation_lrt_asymptotic_bootstrap_comparison.csv"), row.names = FALSE)

  sim_lrt_long_tex <- data.frame(
    Dataset = lrt_comp$Dataset,
    Model = latex_escape(lrt_comp$RestrictedModel),
    Reference = lrt_comp$Reference,
    `Asym. reject` = fmt(lrt_comp$AsymRejectRate, 2),
    `Boot. reject` = fmt(lrt_comp$BootRejectRate, 2),
    `Mean asym. p` = fmt_p(lrt_comp$MeanAsymP),
    `Mean boot. p` = fmt_p(lrt_comp$MeanBootP),
    check.names = FALSE
  )
  write_simple_tex(
    file.path(tex_dir, "simulation_lrt_asymptotic_bootstrap_comparison.tex"),
    "Comparison of asymptotic/chi-bar and parametric-bootstrap LRT rejection rates in the simulation study. Bootstrap entries use 20 simulation replicates and 99 bootstrap samples per replicate.",
    "tab:statscomp_sim_lrt_asym_boot",
    names(sim_lrt_long_tex),
    sim_lrt_long_tex,
    align = "lllrrrr",
    size = "\\scriptsize"
  )

  sim_keys <- unique(lrt_comp[c("Dataset", "Scenario")])
  sim_wide <- sim_keys[order(sim_keys$Scenario), ]
  for (m in model_order) {
    vals <- character(nrow(sim_wide))
    for (i in seq_len(nrow(sim_wide))) {
      d <- lrt_comp[lrt_comp$Scenario == sim_wide$Scenario[i] & lrt_comp$RestrictedModel == m, ]
      cell <- paste0(fmt(d$AsymRejectRate[1], 2), "/", fmt(d$BootRejectRate[1], 2))
      if (!is.na(d$AsymRejectRate[1]) && !is.na(d$BootRejectRate[1]) && abs(d$AsymRejectRate[1] - d$BootRejectRate[1]) > 1e-9) {
        cell <- paste0("\\textbf{", cell, "}")
      }
      vals[i] <- cell
    }
    sim_wide[[m]] <- vals
  }
  write.csv(sim_wide, file.path(out_dir, "simulation_lrt_asymptotic_bootstrap_matrix.csv"), row.names = FALSE)
  sim_wide_tex <- sim_wide[c("Dataset", model_order)]
  write_simple_tex(
    file.path(tex_dir, "simulation_lrt_asymptotic_bootstrap_matrix.tex"),
    "Simulation LRT rejection rates comparing asymptotic/chi-bar calibration with parametric-bootstrap calibration. Each cell reports asymptotic/bootstrap rejection rates at the 5\\% level; bold marks a difference between the two calibrations.",
    "tab:statscomp_sim_lrt_matrix",
    c("Dataset", model_order),
    sim_wide_tex,
    align = "lcccccc",
    size = "\\scriptsize"
  )
}

# Real bootstrap LRT decision matrix.
real_boot$Decision <- ifelse(real_boot$BootstrapPValue < 0.05, "Reject", "Fail")
real_long <- real_boot[order(real_boot$Dataset, real_boot$Link, match(real_boot$RestrictedModel, model_order)), ]
write.csv(real_long, file.path(out_dir, "real_bootstrap_lrt_long.csv"), row.names = FALSE)
real_keys <- unique(real_long[c("Dataset", "Link")])
real_wide <- real_keys
for (m in model_order) {
  vals <- character(nrow(real_keys))
  for (i in seq_len(nrow(real_keys))) {
    d <- real_long[real_long$Dataset == real_keys$Dataset[i] & real_long$Link == real_keys$Link[i] & real_long$RestrictedModel == m, ]
    cell <- paste0(fmt_p(d$BootstrapPValue[1]), " (", ifelse(d$Decision[1] == "Reject", "R", "F"), ")")
    if (d$Decision[1] == "Reject") cell <- paste0("\\textbf{", cell, "}")
    vals[i] <- cell
  }
  real_wide[[m]] <- vals
}
write.csv(real_wide, file.path(out_dir, "real_bootstrap_lrt_decision_matrix.csv"), row.names = FALSE)
real_tex <- real_wide
real_tex$Dataset <- latex_dataset(real_tex$Dataset)
real_tex$Link <- latex_link(real_tex$Link)
write_simple_tex(
  file.path(tex_dir, "real_bootstrap_lrt_decision_matrix.tex"),
  "Parametric-bootstrap LRT p-values and decisions for the real datasets. R denotes rejection of the restricted model at the 5\\% level and F denotes failure to reject. Bold entries indicate rejection. All real-data entries use 99 bootstrap samples.",
  "tab:statscomp_real_bootstrap_lrt",
  c("Dataset", "Link", model_order),
  real_tex,
  align = "llcccccc",
  size = "\\tiny"
)

# Real-data asymptotic/chi-bar versus parametric-bootstrap LRT comparison.
real_asym_file <- file.path(root, "outputs", "real_analytical", "tables", "lrt_results_detailed.csv")
if (file.exists(real_asym_file)) {
  real_asym <- read.csv(real_asym_file, stringsAsFactors = FALSE)
  real_asym$Reference <- ifelse(!is.na(real_asym$boundary_dim) & real_asym$boundary_dim > 0, "chi-bar", "chi-square")
  real_comp <- merge(
    real_asym,
    real_long[c("Dataset", "Link", "RestrictedModel", "ObservedLRT", "BootstrapPValue", "BootstrapRepsSucceeded", "Decision")],
    by = c("Dataset", "Link", "RestrictedModel"),
    suffixes = c("Asym", "Boot"),
    all.x = TRUE
  )
  real_comp <- real_comp[order(real_comp$Dataset, real_comp$Link, match(real_comp$RestrictedModel, model_order)), ]
  write.csv(real_comp, file.path(out_dir, "real_lrt_asymptotic_bootstrap_full.csv"), row.names = FALSE)

  real_comp_keys <- unique(real_comp[c("Dataset", "Link")])
  real_comp_wide <- real_comp_keys
  for (m in model_order) {
    vals <- character(nrow(real_comp_keys))
    for (i in seq_len(nrow(real_comp_keys))) {
      d <- real_comp[real_comp$Dataset == real_comp_keys$Dataset[i] & real_comp$Link == real_comp_keys$Link[i] & real_comp$RestrictedModel == m, ]
      asym_short <- ifelse(d$decision[1] == "Reject", "R", "F")
      boot_short <- ifelse(d$Decision[1] == "Reject", "R", "F")
      cell <- paste0(asym_short, "/", boot_short)
      if (asym_short != boot_short) cell <- paste0("\\textbf{", cell, "}")
      vals[i] <- cell
    }
    real_comp_wide[[m]] <- vals
  }
  write.csv(real_comp_wide, file.path(out_dir, "real_lrt_asymptotic_bootstrap_matrix.csv"), row.names = FALSE)
  real_comp_wide_tex <- real_comp_wide
  real_comp_wide_tex$Dataset <- latex_dataset(real_comp_wide_tex$Dataset)
  real_comp_wide_tex$Link <- latex_link(real_comp_wide_tex$Link)
  write_simple_tex(
    file.path(tex_dir, "real_lrt_asymptotic_bootstrap_matrix.tex"),
    "Real-data LRT decision comparison. Each cell reports asymptotic/chi-bar decision followed by parametric-bootstrap decision at the 5\\% level, with R denoting reject and F denoting fail to reject; bold cells mark disagreement. The bootstrap calibration uses 99 samples for every real-data comparison.",
    "tab:statscomp_real_lrt_matrix",
    c("Dataset", "Link", model_order),
    real_comp_wide_tex,
    align = "llcccccc",
    size = "\\tiny"
  )

  real_comp_tex <- data.frame(
    Dataset = latex_dataset(real_comp$Dataset),
    Link = latex_link(real_comp$Link),
    Model = latex_escape(real_comp$RestrictedModel),
    Reference = real_comp$Reference,
    LRT = fmt(real_comp$LRT, 2),
    `Asym. p` = fmt_p(real_comp$p_value),
    `Asym. dec.` = real_comp$decision,
    `Boot. p` = fmt_p(real_comp$BootstrapPValue),
    `Boot. dec.` = real_comp$Decision,
    B = real_comp$BootstrapRepsSucceeded,
    check.names = FALSE
  )
  write_simple_tex(
    file.path(tex_dir, "real_lrt_asymptotic_bootstrap_full.tex"),
    "Real-data LRT p-values under asymptotic/chi-bar calibration and parametric-bootstrap calibration. The bootstrap column uses 99 samples for every comparison.",
    "tab:supp_real_lrt_asym_boot_full",
    names(real_comp_tex),
    real_comp_tex,
    align = "lllrrrrrrr",
    size = "\\tiny"
  )
}

# Real cross-validated log score summary.
score_col <- if ("mean_mean_test_log_score" %in% names(real_cv)) "mean_mean_test_log_score" else grep("mean.*test.*score", names(real_cv), value = TRUE)[1]
cv_summary <- do.call(rbind, lapply(split(real_cv, paste(real_cv$Dataset, real_cv$Link)), function(d) {
  d <- d[order(-d[[score_col]]), ]
  full <- d[d$Model == "ZIP--ZINB", ]
  data.frame(
    Dataset = d$Dataset[1],
    Link = d$Link[1],
    BestModel = d$Model[1],
    BestScore = d[[score_col]][1],
    FullScore = full[[score_col]][1],
    DeltaBestMinusFull = d[[score_col]][1] - full[[score_col]][1],
    FullRank = which(d$Model == "ZIP--ZINB")
  )
}))
cv_summary <- cv_summary[order(cv_summary$Dataset, cv_summary$Link), ]
write.csv(cv_summary, file.path(out_dir, "real_cv_log_score_summary.csv"), row.names = FALSE)
cv_tex <- data.frame(
  Dataset = latex_dataset(cv_summary$Dataset),
  Link = latex_link(cv_summary$Link),
  `Best model` = latex_escape(cv_summary$BestModel),
  `Best score` = fmt(cv_summary$BestScore, 4),
  `ZIP--ZINB score` = fmt(cv_summary$FullScore, 4),
  `Difference` = fmt(cv_summary$DeltaBestMinusFull, 5),
  `Full rank` = cv_summary$FullRank,
  check.names = FALSE
)
write_simple_tex(
  file.path(tex_dir, "real_cv_log_score_summary.tex"),
  "Five-fold out-of-sample mean log-score summary for the real datasets. Larger values indicate better predictive performance; the difference column is best score minus ZIP--ZINB score.",
  "tab:statscomp_real_cv_log_scores",
  names(cv_tex),
  cv_tex,
  align = "llcrrrr",
  size = "\\scriptsize"
)

# Real diagnostic summary.
real_zero$AbsError <- abs(real_zero$ObservedZeroRate - real_zero$FittedZeroProbability)
real_tail$AbsRateError <- abs(real_tail$ObservedTailRate - real_tail$ExpectedTailRate)
zero_mae <- aggregate(AbsError ~ Dataset + Link, real_zero, mean)
tail_max <- aggregate(AbsRateError ~ Dataset + Link, real_tail, max)
real_diag <- merge(real_rqr, zero_mae, by = c("Dataset", "Link"))
real_diag <- merge(real_diag, tail_max, by = c("Dataset", "Link"))
real_diag <- real_diag[order(real_diag$Dataset, real_diag$Link), ]
write.csv(real_diag, file.path(out_dir, "real_diagnostic_summary.csv"), row.names = FALSE)
real_diag_tex <- data.frame(
  Dataset = latex_dataset(real_diag$Dataset),
  Link = latex_link(real_diag$Link),
  `RQR mean` = fmt(real_diag$mean, 3),
  `RQR SD` = fmt(real_diag$sd, 3),
  `Zero MAE` = fmt(real_diag$AbsError, 4),
  `Max tail error` = fmt(real_diag$AbsRateError, 4),
  check.names = FALSE
)
write_simple_tex(
  file.path(tex_dir, "real_diagnostic_summary.tex"),
  "Real-data diagnostic summary for the fitted ZIP--ZINB model. The zero MAE is the mean absolute difference between observed and fitted zero proportions across dose groups; the tail error is the maximum absolute tail-rate discrepancy across the thresholds considered.",
  "tab:statscomp_real_diagnostics",
  names(real_diag_tex),
  real_diag_tex,
  align = "llrrrr",
  size = "\\scriptsize"
)

# Synthetic diagnostic summary.
syn_zero$AbsError <- abs(syn_zero$ObservedZeroRate - syn_zero$FittedZeroProbability)
syn_tail$AbsRateError <- abs(syn_tail$ObservedTailRate - syn_tail$ExpectedTailRate)
syn_rqr_summary <- aggregate(cbind(mean, sd) ~ Scenario + Dataset, syn_rqr, mean)
syn_zero_mae <- aggregate(AbsError ~ Scenario + Dataset, syn_zero, mean)
syn_tail_max <- aggregate(AbsRateError ~ Scenario + Dataset, syn_tail, max)
score_col2 <- if ("mean_mean_test_log_score" %in% names(syn_log)) "mean_mean_test_log_score" else grep("mean.*test.*score", names(syn_log), value = TRUE)[1]
syn_best <- do.call(rbind, lapply(split(syn_log, syn_log$Scenario), function(d) d[which.max(d[[score_col2]]), ]))
syn_diag <- merge(syn_rqr_summary, syn_zero_mae, by = c("Scenario", "Dataset"))
syn_diag <- merge(syn_diag, syn_tail_max, by = c("Scenario", "Dataset"))
syn_diag <- merge(syn_diag, syn_best[c("Scenario", "Model", score_col2)], by = "Scenario")
syn_diag <- syn_diag[order(syn_diag$Scenario), ]
write.csv(syn_diag, file.path(out_dir, "synthetic_diagnostic_summary.csv"), row.names = FALSE)
syn_diag_tex <- data.frame(
  Scenario = syn_diag$Scenario,
  `Best log-score model` = latex_escape(syn_diag$Model),
  `Mean log score` = fmt(syn_diag[[score_col2]], 4),
  `RQR mean` = fmt(syn_diag$mean, 3),
  `RQR SD` = fmt(syn_diag$sd, 3),
  `Zero MAE` = fmt(syn_diag$AbsError, 4),
  `Max tail error` = fmt(syn_diag$AbsRateError, 4),
  check.names = FALSE
)
write_simple_tex(
  file.path(tex_dir, "synthetic_diagnostic_summary.tex"),
  "Synthetic diagnostic summary for the fitted ZIP--ZINB model.",
  "tab:statscomp_synthetic_diagnostics",
  names(syn_diag_tex),
  syn_diag_tex,
  align = "lcrrrrr",
  size = "\\tiny"
)

# Runtime summaries.
syn_runtime <- do.call(rbind, lapply(split(syn_boot, paste(syn_boot$Scenario, syn_boot$RestrictedModel, sep = "|")), function(d) {
  data.frame(Scenario = d$Scenario[1], RestrictedModel = d$RestrictedModel[1], MeanBootstrapMinutes = mean(d$ElapsedSec) / 60, MaxBootstrapMinutes = max(d$ElapsedSec) / 60, Reps = nrow(d), B = min(d$BootstrapRepsSucceeded))
}))
syn_runtime <- syn_runtime[order(syn_runtime$Scenario, match(syn_runtime$RestrictedModel, model_order)), ]
write.csv(syn_runtime, file.path(out_dir, "synthetic_bootstrap_runtime_summary.csv"), row.names = FALSE)
syn_boot_runtime_main <- do.call(rbind, lapply(split(syn_boot, syn_boot$Scenario), function(d) {
  data.frame(
    Scenario = d$Scenario[1],
    Dataset = d$Dataset[1],
    MedianBootstrapMinutes = median(d$ElapsedSec) / 60,
    MeanBootstrapMinutes = mean(d$ElapsedSec) / 60,
    MaxBootstrapMinutes = max(d$ElapsedSec) / 60,
    Units = nrow(d),
    B = min(d$BootstrapRepsSucceeded)
  )
}))
syn_boot_runtime_main <- syn_boot_runtime_main[order(syn_boot_runtime_main$Scenario), ]
write.csv(syn_boot_runtime_main, file.path(out_dir, "synthetic_bootstrap_runtime_summary_main.csv"), row.names = FALSE)
syn_boot_runtime_tex <- data.frame(
  Dataset = syn_boot_runtime_main$Dataset,
  `Median min.` = fmt(syn_boot_runtime_main$MedianBootstrapMinutes, 2),
  `Mean min.` = fmt(syn_boot_runtime_main$MeanBootstrapMinutes, 2),
  `Max min.` = fmt(syn_boot_runtime_main$MaxBootstrapMinutes, 2),
  `Bootstrap units` = syn_boot_runtime_main$Units,
  B = syn_boot_runtime_main$B,
  check.names = FALSE
)
write_simple_tex(
  file.path(tex_dir, "synthetic_bootstrap_runtime_summary_main.tex"),
  "Synthetic parametric-bootstrap runtime by scenario. A bootstrap unit is one restricted-model comparison in one simulation replicate; each unit uses 99 bootstrap samples.",
  "tab:statscomp_sim_boot_runtime",
  names(syn_boot_runtime_tex),
  syn_boot_runtime_tex,
  align = "lrrrrr",
  size = "\\scriptsize"
)

real_runtime <- real_long
real_runtime$BootstrapMinutes <- real_runtime$ElapsedSec / 60
real_runtime <- real_runtime[order(real_runtime$Dataset, real_runtime$Link, match(real_runtime$RestrictedModel, model_order)), c("Dataset", "Link", "RestrictedModel", "BootstrapMinutes", "BootstrapRepsSucceeded")]
write.csv(real_runtime, file.path(out_dir, "real_bootstrap_runtime_summary.csv"), row.names = FALSE)
real_boot_runtime_main <- do.call(rbind, lapply(split(real_runtime, paste(real_runtime$Dataset, real_runtime$Link, sep = "|")), function(d) {
  data.frame(
    Dataset = d$Dataset[1],
    Link = d$Link[1],
    MedianBootstrapMinutes = median(d$BootstrapMinutes),
    MaxBootstrapMinutes = max(d$BootstrapMinutes),
    Comparisons = nrow(d),
    B = min(d$BootstrapRepsSucceeded)
  )
}))
real_boot_runtime_main <- real_boot_runtime_main[order(real_boot_runtime_main$Dataset, real_boot_runtime_main$Link), ]

runtime_overall <- data.frame(
  Analysis = c("Synthetic bootstrap LRT", "Real bootstrap LRT", "Real diagnostics", "Real five-fold CV"),
  Unit = c("scenario/model/rep", "dataset/link/model", "all real data", "all real folds"),
  `Median minutes` = c(median(syn_boot$ElapsedSec) / 60, median(real_long$ElapsedSec) / 60, NA, NA),
  `Maximum minutes` = c(max(syn_boot$ElapsedSec) / 60, max(real_long$ElapsedSec) / 60, 0.50, 0.48),
  `Bootstrap B` = c(99, 99, NA, NA),
  check.names = FALSE
)
write.csv(runtime_overall, file.path(out_dir, "runtime_overall_summary.csv"), row.names = FALSE)
runtime_tex <- data.frame(
  Analysis = runtime_overall$Analysis,
  Unit = runtime_overall$Unit,
  `Median min.` = fmt(runtime_overall$`Median minutes`, 2),
  `Max min.` = fmt(runtime_overall$`Maximum minutes`, 2),
  `Bootstrap B` = ifelse(is.na(runtime_overall$`Bootstrap B`), "--", runtime_overall$`Bootstrap B`),
  check.names = FALSE
)
write_simple_tex(
  file.path(tex_dir, "runtime_overall_summary.tex"),
  "Runtime summary for the extended robustness analyses.",
  "tab:statscomp_runtime_summary",
  names(runtime_tex),
  runtime_tex,
  align = "llrrr",
  size = "\\scriptsize"
)

# Simulation coverage, log-score, and runtime summaries from the full extended
# simulation run. These are generated separately from the bootstrap robustness
# outputs so the summary tables can report observed-information uncertainty and
# computational cost in a compact, reproducible way.
sim_dir <- file.path(root, "outputs", "sim_analytical", "statscomp_full_extended", "tables")
if (dir.exists(sim_dir)) {
  sim_cov <- read.csv(file.path(sim_dir, "simulation_coverage_summary.csv"), stringsAsFactors = FALSE)
  sim_cov$Block <- ifelse(grepl("beta", sim_cov$Parameter), "$\\beta$",
                   ifelse(grepl("gamma", sim_cov$Parameter), "$\\gamma$", "Scalar"))
  cov_compact <- do.call(rbind, lapply(split(sim_cov, paste(sim_cov$Dataset, sim_cov$Block, sep = "|")), function(d) {
    data.frame(
      Dataset = d$Dataset[1],
      Block = d$Block[1],
      MinCoverage = min(d$coverage_rate),
      MedianCoverage = median(d$coverage_rate),
      MedianMeanSE = median(d$mean_se),
      MaxMeanSE = max(d$mean_se),
      N = min(d$n_intervals)
    )
  }))
  cov_compact <- cov_compact[order(cov_compact$Dataset, match(cov_compact$Block, c("Scalar", "$\\beta$", "$\\gamma$"))), ]
  write.csv(cov_compact, file.path(out_dir, "simulation_coverage_compact.csv"), row.names = FALSE)
  cov_compact_tex <- data.frame(
    Dataset = cov_compact$Dataset,
    Block = cov_compact$Block,
    `Min coverage` = fmt(cov_compact$MinCoverage, 2),
    `Median coverage` = fmt(cov_compact$MedianCoverage, 2),
    `Median mean SE` = fmt(cov_compact$MedianMeanSE, 3),
    `Max mean SE` = fmt(cov_compact$MaxMeanSE, 3),
    check.names = FALSE
  )
  write_simple_tex(
    file.path(tex_dir, "simulation_coverage_compact.tex"),
    "Observed-information uncertainty calibration in the simulation study. Coverage is the empirical coverage of nominal 95\\% Wald intervals across 100 simulation replicates; mean SE is the average model-based standard error within a parameter block.",
    "tab:statscomp_sim_coverage",
    names(cov_compact_tex),
    cov_compact_tex,
    align = "llrrrr",
    size = "\\scriptsize",
    resize = FALSE
  )

  param_latex <- c(
    rho = "$\\rho$", alpha = "$\\alpha$", logit_rho = "$\\operatorname{logit}(\\rho)$", log_alpha = "$\\log(\\alpha)$",
    "beta_z[0]" = "$\\beta_{10}$", "beta_z[1]" = "$\\beta_{11}$",
    "gamma_z[0]" = "$\\gamma_{10}$", "gamma_z[1]" = "$\\gamma_{11}$",
    "beta_nb[0]" = "$\\beta_{20}$", "beta_nb[1]" = "$\\beta_{21}$",
    "gamma_nb[0]" = "$\\gamma_{20}$", "gamma_nb[1]" = "$\\gamma_{21}$"
  )
  sim_cov$ParameterLatex <- ifelse(sim_cov$Parameter %in% names(param_latex), param_latex[sim_cov$Parameter], sim_cov$Parameter)
  sim_cov <- sim_cov[order(sim_cov$Dataset, match(sim_cov$Parameter, names(param_latex))), ]
  write.csv(sim_cov, file.path(out_dir, "simulation_coverage_full.csv"), row.names = FALSE)
  cov_full_tex <- data.frame(
    Dataset = sim_cov$Dataset,
    Parameter = sim_cov$ParameterLatex,
    Scale = gsub("_", "\\\\_", sim_cov$Scale),
    Coverage = fmt(sim_cov$coverage_rate, 2),
    `Mean SE` = fmt(sim_cov$mean_se, 4),
    `Intervals` = sim_cov$n_intervals,
    check.names = FALSE
  )
  write_simple_tex(
    file.path(tex_dir, "simulation_coverage_full.tex"),
    "Parameter-level observed-information coverage summary for nominal 95\\% Wald intervals in the simulation study.",
    "tab:supp_sim_coverage_full",
    names(cov_full_tex),
    cov_full_tex,
    align = "lllrrr",
    size = "\\scriptsize"
  )

  sim_scores <- read.csv(file.path(sim_dir, "simulation_test_log_scores.csv"), stringsAsFactors = FALSE)
  sim_scores <- sim_scores[order(sim_scores$Dataset, match(sim_scores$Model, c("Poisson", "NB", "ZIP", "ZINB", "P+NB", "Z+P+NB", "ZIP--ZINB"))), ]
  write.csv(sim_scores, file.path(out_dir, "simulation_test_log_scores.csv"), row.names = FALSE)
  sim_scores_tex <- data.frame(
    Dataset = sim_scores$Dataset,
    Model = latex_escape(sim_scores$Model),
    `Mean test log score` = fmt(sim_scores$mean_test_log_score, 4),
    SD = fmt(sim_scores$sd_test_log_score, 4),
    `Reps` = sim_scores$n_reps,
    check.names = FALSE
  )
  write_simple_tex(
    file.path(tex_dir, "simulation_test_log_scores.tex"),
    "Out-of-sample log scores for the simulation study. Larger values indicate better predictive performance.",
    "tab:supp_sim_test_log_scores",
    names(sim_scores_tex),
    sim_scores_tex,
    align = "llrrr",
    size = "\\scriptsize"
  )

  sim_runtime <- read.csv(file.path(sim_dir, "simulation_runtime_summary.csv"), stringsAsFactors = FALSE)
  sim_runtime <- sim_runtime[order(sim_runtime$Dataset, match(sim_runtime$Model, c("Poisson", "NB", "ZIP", "ZINB", "P+NB", "Z+P+NB", "ZIP--ZINB"))), ]
  write.csv(sim_runtime, file.path(out_dir, "simulation_runtime_by_model.csv"), row.names = FALSE)
  sim_runtime_tex <- data.frame(
    Dataset = sim_runtime$Dataset,
    Model = latex_escape(sim_runtime$Model),
    `Mean sec.` = fmt(sim_runtime$mean_sec, 2),
    `Median sec.` = fmt(sim_runtime$median_sec, 2),
    `SD sec.` = fmt(sim_runtime$sd_sec, 2),
    `Fits` = sim_runtime$n_fits,
    check.names = FALSE
  )
  write_simple_tex(
    file.path(tex_dir, "simulation_runtime_by_model.tex"),
    "Per-fit runtime summary for the simulation model comparisons.",
    "tab:supp_sim_runtime_by_model",
    names(sim_runtime_tex),
    sim_runtime_tex,
    align = "llrrrr",
    size = "\\scriptsize"
  )

  sim_raw_runtime_file <- file.path(root, "outputs", "sim_analytical", "statscomp_full_extended", "raw_runtime.csv")
  if (file.exists(sim_raw_runtime_file)) {
    sim_raw_runtime <- read.csv(sim_raw_runtime_file, stringsAsFactors = FALSE)
    sim_rep_runtime <- aggregate(fit_elapsed_sec ~ Scenario + Dataset + Rep, sim_raw_runtime, sum)
    names(sim_rep_runtime)[names(sim_rep_runtime) == "fit_elapsed_sec"] <- "TotalFitSeconds"
    write.csv(sim_rep_runtime, file.path(out_dir, "simulation_runtime_by_mc_replicate.csv"), row.names = FALSE)
    sim_rep_runtime_summary <- do.call(rbind, lapply(split(sim_rep_runtime, sim_rep_runtime$Scenario), function(d) {
      data.frame(
        Scenario = d$Scenario[1],
        Dataset = d$Dataset[1],
        MeanSeconds = mean(d$TotalFitSeconds),
        MedianSeconds = median(d$TotalFitSeconds),
        MaxSeconds = max(d$TotalFitSeconds),
        Reps = nrow(d)
      )
    }))
    sim_rep_runtime_summary <- sim_rep_runtime_summary[order(sim_rep_runtime_summary$Scenario), ]
    write.csv(sim_rep_runtime_summary, file.path(out_dir, "simulation_mc_replicate_runtime_summary.csv"), row.names = FALSE)
    sim_rep_runtime_tex <- data.frame(
      Dataset = sim_rep_runtime_summary$Dataset,
      `Mean sec.` = fmt(sim_rep_runtime_summary$MeanSeconds, 2),
      `Median sec.` = fmt(sim_rep_runtime_summary$MedianSeconds, 2),
      `Max sec.` = fmt(sim_rep_runtime_summary$MaxSeconds, 2),
      `Replicates` = sim_rep_runtime_summary$Reps,
      check.names = FALSE
    )
    write_simple_tex(
      file.path(tex_dir, "simulation_mc_replicate_runtime_summary.tex"),
      "Runtime per synthetic Monte Carlo replicate, summing the successful fits of all candidate models within each replicate.",
      "tab:statscomp_sim_mc_runtime",
      names(sim_rep_runtime_tex),
      sim_rep_runtime_tex,
      align = "lrrrr",
      size = "\\scriptsize"
    )
  }
}

real_fit_runtime_file <- file.path(rob, "statscomp_real_diagnostics", "raw_runtime.csv")
if (file.exists(real_fit_runtime_file) && exists("real_boot_runtime_main")) {
  real_fit_runtime <- read.csv(real_fit_runtime_file, stringsAsFactors = FALSE)
  real_fit_runtime_main <- do.call(rbind, lapply(split(real_fit_runtime, paste(real_fit_runtime$Dataset, real_fit_runtime$Link, sep = "|")), function(d) {
    data.frame(
      Dataset = d$Dataset[1],
      Link = d$Link[1],
      MedianFitSeconds = median(d$fit_elapsed_sec, na.rm = TRUE),
      MaxFitSeconds = max(d$fit_elapsed_sec, na.rm = TRUE)
    )
  }))
  real_runtime_main <- merge(real_fit_runtime_main, real_boot_runtime_main, by = c("Dataset", "Link"), all = TRUE)
  real_runtime_main <- real_runtime_main[order(real_runtime_main$Dataset, real_runtime_main$Link), ]
  write.csv(real_runtime_main, file.path(out_dir, "real_runtime_summary_main.csv"), row.names = FALSE)
  real_runtime_main_tex <- data.frame(
    Dataset = latex_dataset(real_runtime_main$Dataset),
    Link = latex_link(real_runtime_main$Link),
    `Median fit sec.` = fmt(real_runtime_main$MedianFitSeconds, 2),
    `Max fit sec.` = fmt(real_runtime_main$MaxFitSeconds, 2),
    `Median boot. min.` = fmt(real_runtime_main$MedianBootstrapMinutes, 2),
    `Max boot. min.` = fmt(real_runtime_main$MaxBootstrapMinutes, 2),
    B = real_runtime_main$B,
    check.names = FALSE
  )
  write_simple_tex(
    file.path(tex_dir, "real_runtime_summary_main.tex"),
    "Real-data runtime summary. Fit times are ordinary candidate-model fits; bootstrap times are per nested-model comparison and use 99 bootstrap samples for every real-data entry.",
    "tab:statscomp_real_runtime",
    names(real_runtime_main_tex),
    real_runtime_main_tex,
    align = "llrrrrr",
    size = "\\scriptsize"
  )
}

main_diag_panel <- c(
  "\\begin{figure}[!htbp]",
  "\\centering",
  "\\begin{minipage}{0.48\\textwidth}",
  "\\centering",
  "\\includegraphics[width=\\linewidth]{generated_figures/robustness/rootogram_H2AX1H60_log.png}\\\\[-2pt]",
  "{\\scriptsize (a) Rootogram, \\texttt{1H60}, log link}",
  "\\end{minipage}\\hfill",
  "\\begin{minipage}{0.48\\textwidth}",
  "\\centering",
  "\\includegraphics[width=\\linewidth]{generated_figures/robustness/zero_probability_H2AX1H60_log.png}\\\\[-2pt]",
  "{\\scriptsize (b) Zero probabilities, \\texttt{1H60}, log link}",
  "\\end{minipage}",
  "",
  "\\vspace{0.6em}",
  "",
  "\\begin{minipage}{0.48\\textwidth}",
  "\\centering",
  "\\includegraphics[width=\\linewidth]{generated_figures/robustness/rootogram_H2AX4H100_log.png}\\\\[-2pt]",
  "{\\scriptsize (c) Rootogram, \\texttt{4H100}, log link}",
  "\\end{minipage}\\hfill",
  "\\begin{minipage}{0.48\\textwidth}",
  "\\centering",
  "\\includegraphics[width=\\linewidth]{generated_figures/robustness/zero_probability_H2AX4H100_log.png}\\\\[-2pt]",
  "{\\scriptsize (d) Zero probabilities, \\texttt{4H100}, log link}",
  "\\end{minipage}",
  "\\caption{Representative real-data diagnostics for the fitted ZIP--ZINB model. The rootograms compare observed and fitted frequencies, while the zero-probability plots compare observed and fitted zero rates across dose groups.}",
  "\\label{fig:statscomp_real_diag_panel}",
  "\\end{figure}",
  ""
)
writeLines(main_diag_panel, file.path(tex_dir, "main_real_diagnostic_panel.tex"))

make_diag_figure <- function(files, caption, label) {
  c(
    "\\begin{figure}[!htbp]",
    "\\centering",
    paste0("\\begin{minipage}{0.32\\textwidth}\\centering\\includegraphics[width=\\linewidth]{", files[1], "}\\\\[-2pt]{\\scriptsize fitted frequencies}\\end{minipage}\\hfill"),
    paste0("\\begin{minipage}{0.32\\textwidth}\\centering\\includegraphics[width=\\linewidth]{", files[2], "}\\\\[-2pt]{\\scriptsize hanging rootogram}\\end{minipage}\\hfill"),
    paste0("\\begin{minipage}{0.32\\textwidth}\\centering\\includegraphics[width=\\linewidth]{", files[3], "}\\\\[-2pt]{\\scriptsize zero probabilities}\\end{minipage}"),
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\end{figure}",
    ""
  )
}

real_figs <- list(
  C2_identity = c("fitted_frequency_C2_identity.png", "rootogram_C2_identity.png", "zero_probability_C2_identity.png"),
  C2_log = c("fitted_frequency_C2_log.png", "rootogram_C2_log.png", "zero_probability_C2_log.png"),
  C3_identity = c("fitted_frequency_C3_identity.png", "rootogram_C3_identity.png", "zero_probability_C3_identity.png"),
  C3_log = c("fitted_frequency_C3_log.png", "rootogram_C3_log.png", "zero_probability_C3_log.png"),
  H2AX1H60_identity = c("fitted_frequency_H2AX1H60_identity.png", "rootogram_H2AX1H60_identity.png", "zero_probability_H2AX1H60_identity.png"),
  H2AX1H60_log = c("fitted_frequency_H2AX1H60_log.png", "rootogram_H2AX1H60_log.png", "zero_probability_H2AX1H60_log.png"),
  H2AX4H100_identity = c("fitted_frequency_H2AX4H100_identity.png", "rootogram_H2AX4H100_identity.png", "zero_probability_H2AX4H100_identity.png"),
  H2AX4H100_log = c("fitted_frequency_H2AX4H100_log.png", "rootogram_H2AX4H100_log.png", "zero_probability_H2AX4H100_log.png")
)
real_frag <- unlist(Map(function(nm, f) {
  bits <- strsplit(nm, "_")[[1]]
  link <- tail(bits, 1)
  dataset <- paste(head(bits, -1), collapse = "_")
  make_diag_figure(
    file.path("generated_figures", "robustness", f),
    paste0("Real-data diagnostic plots for ", latex_escape(dataset), " under the ", link, " link."),
    paste0("fig:supp_real_diag_", gsub("[^A-Za-z0-9]+", "_", nm))
  )
}, names(real_figs), real_figs))
writeLines(real_frag, file.path(tex_dir, "supplement_real_diagnostic_figures.tex"))

syn_figs <- list()
for (sc in paste0("D", 1:4)) {
  for (rep in 0:2) {
    nm <- paste0(sc, "_rep", rep)
    syn_figs[[nm]] <- c(paste0("fitted_frequency_", nm, ".png"), paste0("rootogram_", nm, ".png"), paste0("zero_probability_", nm, ".png"))
  }
}
syn_frag <- unlist(Map(function(nm, f) {
  make_diag_figure(
    file.path("generated_figures", "robustness", f),
    paste0("Synthetic diagnostic plots for ", sub("_", ", ", nm), "."),
    paste0("fig:supp_syn_diag_", gsub("[^A-Za-z0-9]+", "_", nm))
  )
}, names(syn_figs), syn_figs))
writeLines(syn_frag, file.path(tex_dir, "supplement_synthetic_diagnostic_figures.tex"))

# Copy diagnostic figures into the generated-output folder for easy inclusion.
fig_sources <- c(
  Sys.glob(file.path(rob, "statscomp_real_diagnostics", "figures", "*.png")),
  Sys.glob(file.path(rob, "statscomp_synthetic_diagnostics", "figures", "*.png"))
)
if (length(fig_sources) > 0) {
  invisible(file.copy(fig_sources, file.path(fig_dir, basename(fig_sources)), overwrite = TRUE))
}

manifest <- c(
  "Generated Statistics & Computing robustness tables",
  paste("Date:", Sys.time()),
  paste("CSV output:", out_dir),
  paste("LaTeX output:", tex_dir),
  paste("Copied figures:", fig_dir),
  "",
  "Main generated LaTeX fragments:",
  "synthetic_bootstrap_lrt_summary.tex",
  "simulation_lrt_asymptotic_bootstrap_matrix.tex",
  "simulation_lrt_asymptotic_bootstrap_comparison.tex",
  "real_bootstrap_lrt_decision_matrix.tex",
  "real_lrt_asymptotic_bootstrap_matrix.tex",
  "real_lrt_asymptotic_bootstrap_full.tex",
  "real_cv_log_score_summary.tex",
  "real_diagnostic_summary.tex",
  "synthetic_diagnostic_summary.tex",
  "runtime_overall_summary.tex",
  "simulation_mc_replicate_runtime_summary.tex",
  "synthetic_bootstrap_runtime_summary_main.tex",
  "real_runtime_summary_main.tex",
  "simulation_coverage_compact.tex",
  "simulation_coverage_full.tex",
  "simulation_test_log_scores.tex",
  "simulation_runtime_by_model.tex",
  "main_real_diagnostic_panel.tex",
  "supplement_real_diagnostic_figures.tex",
  "supplement_synthetic_diagnostic_figures.tex"
)
writeLines(manifest, file.path(out_dir, "README_generated_tables.txt"))
writeLines(manifest, file.path(tex_dir, "README_generated_tables.txt"))

cat(paste(manifest, collapse = "\n"), "\n")
