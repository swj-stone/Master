# ---------------------------
# Export a V28 HCC upcoding benchmark
# ---------------------------
# Output structure:
#   OUT_ROOT/
#     baseline_raw.csv
#     baseline_hier.csv
#     baseline_feature_meta.csv
#     target_mapping_table.csv
#     scenario_manifest.csv
#     scenario_low/
#       scenario_spec.csv
#       all_hcc_upcoded_data_pre_hierarchy.csv
#       all_hcc_upcoded_data.csv
#       feature_meta.csv
#       target_audit.csv
#       shift_summary.csv
#       event_summary.csv
#       calibration_log.csv
#     scenario_main/
#       ...
#     scenario_high/
#       ...

library(upcoding)
library(data.table)
library(readr)
library(dplyr)
library(tibble)
library(purrr)
library(stringr)

# ---------------------------
# User-adjustable parameters
# ---------------------------
OUT_ROOT <- "upcoding_v28_top10_ha2025_export"
NUM_PEOPLE <- 5000L
BASELINE_SEED <- 123L
MANIPULATION_SEED <- 456L

NUM_TIMEPOINTS <- 1L
CENSORING_PROP <- 0.0

APPLY_HIERARCHY_BASELINE <- TRUE
APPLY_HIERARCHY_AFTER_UPCODING <- TRUE

UNDERCODING_PROP <- 0.0

SCENARIO_TBL <- tibble(
  scenario = c("scenario_low", "scenario_main", "scenario_high"),
  shift_scale = c(0.50, 0.75, 1.00)
)

HEART_FAILURE_TARGET <- "hcc225"       # Alternatives: "hcc224", "hcc226"
HEART_FAILURE_APPROACH <- "any"        # Alternatives: "lower severity"
CKD_TARGET <- "hcc328"                 # Alternatives: "hcc329"
CKD_APPROACH <- "any"                  # Alternatives: "lower severity"

# Binary-search calibration settings for each target under a single-target run.
CALIBRATION_ITERS <- 12L
CALIBRATION_TOL <- 2e-4

# The upcoding package requires upcoding_prop to lie strictly between 0 and 1.
# We therefore use 0 to mean "skip this target" in our own code and never pass
# 0 or 1 to upcode_all_hccs().
MIN_UPCODING_PROP <- 1e-8
MAX_UPCODING_PROP <- 0.999

# ---------------------------
# Helpers
# ---------------------------
`%||%` <- function(x, y) if (is.null(x)) y else x

fresh_read_dt <- function(path) {
  dt <- fread(path)
  if ("person_id" %in% names(dt)) {
    # The upcoding package samples row IDs using person_id but writes by row index.
    # Resetting person_id to the row order makes this invariant explicit.
    dt[, person_id := seq_len(.N)]
  }
  dt
}

get_hcc_cols <- function(dt) {
  setdiff(names(dt), "person_id")
}

apply_hierarchy_if_needed <- function(dt) {
  apply_v28_cmshcc_hierarchy(copy(dt))
}

write_feature_meta <- function(baseline_dt, manipulated_dt = NULL, scenario_name = "baseline", out_path) {
  hcc_cols <- get_hcc_cols(baseline_dt)
  base_prev <- vapply(hcc_cols, function(h) mean(baseline_dt[[h]]), numeric(1))
  if (is.null(manipulated_dt)) {
    meta <- tibble(
      feature = hcc_cols,
      baseline_prevalence = base_prev,
      manipulated_prevalence = NA_real_,
      prevalence_shift = NA_real_,
      scenario = scenario_name
    )
  } else {
    manip_prev <- vapply(hcc_cols, function(h) mean(manipulated_dt[[h]]), numeric(1))
    meta <- tibble(
      feature = hcc_cols,
      baseline_prevalence = base_prev,
      manipulated_prevalence = manip_prev,
      prevalence_shift = manip_prev - base_prev,
      scenario = scenario_name
    )
  }
  write_csv(meta, out_path)
  invisible(meta)
}

row_sum_summary <- function(dt, label) {
  hcc_cols <- get_hcc_cols(dt)
  rs <- rowSums(as.matrix(dt[, ..hcc_cols]))
  tibble(
    dataset = label,
    n = nrow(dt),
    d = length(hcc_cols),
    mean_active_hccs = mean(rs),
    median_active_hccs = median(rs),
    p75_active_hccs = quantile(rs, 0.75),
    max_active_hccs = max(rs),
    zero_row_fraction = mean(rs == 0),
    nonzero_hcc_columns = sum(colMeans(as.matrix(dt[, ..hcc_cols])) > 0)
  )
}

summarize_shift <- function(baseline_dt, manipulated_dt) {
  hcc_cols <- get_hcc_cols(baseline_dt)
  base_mat <- as.matrix(baseline_dt[, ..hcc_cols])
  manip_mat <- as.matrix(manipulated_dt[, ..hcc_cols])
  row_changes <- rowSums(base_mat != manip_mat)
  tibble(
    frac_rows_changed = mean(row_changes > 0),
    mean_num_feature_changes = mean(row_changes),
    median_num_feature_changes = median(row_changes),
    max_num_feature_changes = max(row_changes)
  )
}

calibrate_single_target <- function(baseline_dt, target_hcc, approach, desired_prev, tmp_parent, seed_base) {
  hcc_cols <- get_hcc_cols(baseline_dt)
  base_prev <- mean(baseline_dt[[target_hcc]])

  if (desired_prev <= base_prev + CALIBRATION_TOL) {
    return(list(prop = 0.0, achieved = base_prev, base_prev = base_prev,
                desired_prev = desired_prev, feasible = TRUE, note = "already_at_or_above_target"))
  }

  eval_prop <- function(prop, iter_id) {
    # Do not pass boundary values to the package validator.
    prop <- min(max(prop, MIN_UPCODING_PROP), MAX_UPCODING_PROP)
    tmp_dir <- file.path(tmp_parent, paste0("calib_", target_hcc, "_", iter_id, "_", sprintf("%.6f", prop)))
    dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)
    spec <- tibble(hcc = target_hcc, approach = approach, upcoding_prop = prop)
    upcode_all_hccs(
      copy(baseline_dt),
      spec,
      out_dir = tmp_dir,
      num_timepoints = NUM_TIMEPOINTS,
      censoring_prop = CENSORING_PROP,
      curr_seed = seed_base + iter_id
    )
    mpath <- file.path(tmp_dir, "all_hcc_upcoded_data.csv")
    mdt <- fresh_read_dt(mpath)
    if (APPLY_HIERARCHY_AFTER_UPCODING) {
      mdt <- apply_hierarchy_if_needed(mdt)
    }
    achieved <- mean(mdt[[target_hcc]])
    unlink(tmp_dir, recursive = TRUE, force = TRUE)
    achieved
  }

  lo <- 0.0
  hi <- MAX_UPCODING_PROP
  best_prop <- hi
  best_achieved <- eval_prop(hi, 999L)
  feasible <- best_achieved >= desired_prev - CALIBRATION_TOL

  if (!feasible) {
    return(list(prop = MAX_UPCODING_PROP, achieved = best_achieved, base_prev = base_prev,
                desired_prev = desired_prev, feasible = FALSE, note = "source_pool_limited_at_cap"))
  }

  for (it in seq_len(CALIBRATION_ITERS)) {
    mid <- (lo + hi) / 2
    achieved <- eval_prop(mid, it)
    if (achieved >= desired_prev) {
      best_prop <- mid
      best_achieved <- achieved
      hi <- mid
    } else {
      lo <- mid
    }
    if (abs(achieved - desired_prev) <= CALIBRATION_TOL) {
      best_prop <- mid
      best_achieved <- achieved
      break
    }
  }
  list(prop = best_prop, achieved = best_achieved, base_prev = base_prev,
       desired_prev = desired_prev, feasible = TRUE, note = "calibrated")
}

# ---------------------------
# Target table: V28 top-10 groups from Kronick et al. 2025
# ---------------------------
# ma_prev and tm_prev are group-level measured prevalences from Exhibit 2.
# For grouped HCCs, target_hcc is the representative upcoding target used in the simulator.
# approach = "lower severity" uses the package's competing-event hierarchy;
# approach = "any" upcodes any eligible row not already coded with the target.
TARGET_TABLE <- tibble(
  target_hcc = c(
    "hcc280",  # COPD / chronic lung disorders
    HEART_FAILURE_TARGET,  # acute or chronic heart failure, except end-stage
    "hcc36",   # diabetes severe/acute complications: severity-based from lower diabetes HCCs
    "hcc48",   # morbid obesity
    "hcc137",  # drug use disorder
    CKD_TARGET,  # CKD stage 3 / moderate CKD
    "hcc155",  # major depression, moderate/severe, without psychosis
    "hcc93",   # rheumatoid arthritis and other specified inflammatory rheumatic disorders
    "hcc253",  # hemiplegia/hemiparesis
    "hcc139"   # alcohol use disorder
  ),
  approach = c(
    "any",
    HEART_FAILURE_APPROACH,
    "lower severity",
    "any",
    "any",
    CKD_APPROACH,
    "any",
    "any",
    "any",
    "any"
  ),
  diagnosis_group = c(
    "COPD, interstitial lung disorders, and other chronic lung disorders",
    "Acute or chronic heart failure, except end-stage",
    "Diabetes with acute or chronic complications",
    "Morbid obesity",
    "Drug use disorder, moderate/severe, or non-psychotic complications",
    "Chronic kidney disease, moderate, stage 3",
    "Major depression, moderate/severe, without psychosis",
    "Rheumatoid arthritis and other specified inflammatory rheumatic disorders",
    "Hemiplegia/hemiparesis",
    "Alcohol use disorder"
  ),
  health_affairs_group_hccs = c(
    "HCC280",
    "HCC224, HCC225, HCC226",
    "HCC36, HCC37, HCC38",
    "HCC48",
    "HCC137",
    "HCC328, HCC329",
    "HCC155",
    "HCC93",
    "HCC253",
    "HCC139"
  ),
  ma_prev = c(0.17, 0.13, 0.31, 0.12, 0.04, 0.14, 0.05, 0.04, 0.02, 0.02),
  tm_prev = c(0.11, 0.10, 0.24, 0.06, 0.02, 0.09, 0.03, 0.03, 0.01, 0.01),
  v28_risk_coefficient = c(0.314, 0.374, 0.175, 0.192, 0.425, 0.134, 0.278, 0.565, 0.385, 0.255),
  removal_effect_percent = c(1.8, 1.3, 1.2, 1.1, 0.9, 0.6, 0.5, 0.5, 0.2, 0.2),
  rationale = c(
    "V28 top-10 group; any-available upcoding is plausible for chronic lung diagnoses.",
    paste0("V28 top-10 group. Default any-available target is ", toupper(HEART_FAILURE_TARGET), "; severity-based heart-failure upcoding was source-pool limited in the sparse baseline."),
    "V28 top-10 group; severity-based upcoding from lower diabetes HCCs is clinically natural.",
    "V28 top-10 group; any-available upcoding.",
    "V28 top-10 group; any-available upcoding because lower-severity source pools may be sparse.",
    paste0("V28 top-10 group. Default any-available target is ", toupper(CKD_TARGET), "; severity-based CKD upcoding was source-pool limited in the sparse baseline."),
    "V28 top-10 group; HCC154 is not included in the new V28 top-10 table, so only HCC155 is targeted.",
    "V28 top-10 group; any-available upcoding.",
    "V28 top-10 group; any-available upcoding.",
    "V28 top-10 group; any-available upcoding."
  )
) %>%
  mutate(prevalence_lift = pmax(ma_prev - tm_prev, 0))

# ---------------------------
# Main export
# ---------------------------
dir.create(OUT_ROOT, showWarnings = FALSE, recursive = TRUE)
write_csv(TARGET_TABLE, file.path(OUT_ROOT, "target_mapping_table.csv"))

# Generate baseline from the raw All-of-Us co-occurrence table, not the dense imputed default.
simulate_baseline_v28_hcc_dt(
  num_people = NUM_PEOPLE,
  reference_cooccurring_hcc_count = cooccurring_v28_hcc_counts_allofus,
  apply_v28_hierarchy = APPLY_HIERARCHY_BASELINE,
  out_dir = OUT_ROOT,
  out_file_prefix = "baseline_raw",
  curr_seed = BASELINE_SEED
)

baseline_raw <- fresh_read_dt(file.path(OUT_ROOT, "baseline_raw.csv"))
if (APPLY_HIERARCHY_BASELINE) {
  baseline_hier <- apply_hierarchy_if_needed(baseline_raw)
} else {
  baseline_hier <- copy(baseline_raw)
}
baseline_hier[, person_id := seq_len(.N)]
fwrite(baseline_hier, file.path(OUT_ROOT, "baseline_hier.csv"))

baseline_for_upcoding <- copy(baseline_hier)
if (UNDERCODING_PROP > 0) {
  baseline_for_upcoding <- undercode_dt(copy(baseline_for_upcoding), undercoding_prop = UNDERCODING_PROP)
  baseline_for_upcoding[, person_id := seq_len(.N)]
}
fwrite(baseline_for_upcoding, file.path(OUT_ROOT, "baseline_tm_like.csv"))
write_feature_meta(baseline_for_upcoding, NULL, "baseline", file.path(OUT_ROOT, "baseline_feature_meta.csv"))
write_csv(row_sum_summary(baseline_for_upcoding, "baseline"), file.path(OUT_ROOT, "baseline_row_sum_summary.csv"))

manifest_rows <- list()

for (rr in seq_len(nrow(SCENARIO_TBL))) {
  scenario_name <- SCENARIO_TBL$scenario[rr]
  shift_scale <- SCENARIO_TBL$shift_scale[rr]
  scenario_dir <- file.path(OUT_ROOT, scenario_name)
  dir.create(scenario_dir, showWarnings = FALSE, recursive = TRUE)

  tmp_parent <- file.path(scenario_dir, "_calibration_tmp")
  dir.create(tmp_parent, showWarnings = FALSE, recursive = TRUE)

  calib_rows <- list()
  spec_rows <- list()
  for (ii in seq_len(nrow(TARGET_TABLE))) {
    row <- TARGET_TABLE[ii, ]
    target <- row$target_hcc
    desired_prev <- mean(baseline_for_upcoding[[target]]) + shift_scale * row$prevalence_lift
    desired_prev <- min(desired_prev, MAX_UPCODING_PROP)
    calib <- calibrate_single_target(
      baseline_dt = baseline_for_upcoding,
      target_hcc = target,
      approach = row$approach,
      desired_prev = desired_prev,
      tmp_parent = tmp_parent,
      seed_base = MANIPULATION_SEED + 1000L * rr + 100L * ii
    )
    calib_rows[[ii]] <- tibble(
      scenario = scenario_name,
      shift_scale = shift_scale,
      target_hcc = target,
      approach = row$approach,
      base_prev = calib$base_prev,
      desired_prev = calib$desired_prev,
      single_target_achieved_prev = calib$achieved,
      calibrated_upcoding_prop = calib$prop,
      feasible_single_target = calib$feasible,
      calibration_note = calib$note,
      diagnosis_group = row$diagnosis_group
    )
    spec_rows[[ii]] <- tibble(
      hcc = target,
      approach = row$approach,
      upcoding_prop = calib$prop
    )
  }
  unlink(tmp_parent, recursive = TRUE, force = TRUE)

  calibration_log <- bind_rows(calib_rows)
  spec_df_full <- bind_rows(spec_rows) %>%
    mutate(
      # Keep the full table for audit, but never pass boundary values to the package.
      upcoding_prop = pmin(pmax(upcoding_prop, 0.0), MAX_UPCODING_PROP),
      pass_to_upcoding = upcoding_prop > MIN_UPCODING_PROP
    )
  spec_df <- spec_df_full %>%
    filter(pass_to_upcoding) %>%
    select(hcc, approach, upcoding_prop)

  write_csv(calibration_log, file.path(scenario_dir, "calibration_log.csv"))
  write_csv(spec_df_full, file.path(scenario_dir, "scenario_spec_full_audit.csv"))
  write_csv(spec_df, file.path(scenario_dir, "scenario_spec.csv"))

  if (nrow(spec_df) > 0) {
    upcode_all_hccs(
      copy(baseline_for_upcoding),
      spec_df,
      out_dir = scenario_dir,
      num_timepoints = NUM_TIMEPOINTS,
      censoring_prop = CENSORING_PROP,
      curr_seed = MANIPULATION_SEED + rr
    )
    manipulated_pre <- fresh_read_dt(file.path(scenario_dir, "all_hcc_upcoded_data.csv"))
  } else {
    manipulated_pre <- copy(baseline_for_upcoding)
  }

  fwrite(manipulated_pre, file.path(scenario_dir, "all_hcc_upcoded_data_pre_hierarchy.csv"))

  if (APPLY_HIERARCHY_AFTER_UPCODING) {
    manipulated_final <- apply_hierarchy_if_needed(manipulated_pre)
  } else {
    manipulated_final <- copy(manipulated_pre)
  }
  manipulated_final[, person_id := seq_len(.N)]
  fwrite(manipulated_final, file.path(scenario_dir, "all_hcc_upcoded_data.csv"))

  write_feature_meta(baseline_for_upcoding, manipulated_final, scenario_name, file.path(scenario_dir, "feature_meta.csv"))
  write_csv(summarize_shift(baseline_for_upcoding, manipulated_final), file.path(scenario_dir, "shift_summary.csv"))
  write_csv(row_sum_summary(manipulated_final, scenario_name), file.path(scenario_dir, "row_sum_summary.csv"))

  target_audit <- TARGET_TABLE %>%
    transmute(
      scenario = scenario_name,
      shift_scale = shift_scale,
      target_hcc,
      approach,
      diagnosis_group,
      health_affairs_group_hccs,
      ma_prev,
      tm_prev,
      prevalence_lift,
      desired_prev = pmin(sapply(target_hcc, function(h) mean(baseline_for_upcoding[[h]])) + shift_scale * prevalence_lift, MAX_UPCODING_PROP),
      baseline_prev = sapply(target_hcc, function(h) mean(baseline_for_upcoding[[h]])),
      final_prev = sapply(target_hcc, function(h) mean(manipulated_final[[h]])),
      final_shift = final_prev - baseline_prev,
      abs_target_error = abs(final_prev - desired_prev)
    )
  write_csv(target_audit, file.path(scenario_dir, "target_audit.csv"))

  event_files <- list.files(scenario_dir, pattern = "_upcoded_data_event_and_time_labels\\.csv$", full.names = TRUE)
  if (length(event_files) > 0) {
    event_summary <- map_dfr(event_files, function(fp) {
      ev <- fread(fp)
      tibble(
        file = basename(fp),
        rows = nrow(ev),
        event_columns = paste(names(ev), collapse = "|")
      )
    })
  } else {
    event_summary <- tibble(file = character(), rows = integer(), event_columns = character())
  }
  write_csv(event_summary, file.path(scenario_dir, "event_summary.csv"))

  manifest_rows[[rr]] <- target_audit %>%
    mutate(num_people = NUM_PEOPLE,
           num_timepoints = NUM_TIMEPOINTS,
           censoring_prop = CENSORING_PROP,
           undercoding_prop = UNDERCODING_PROP,
           apply_hierarchy_baseline = APPLY_HIERARCHY_BASELINE,
           apply_hierarchy_after_upcoding = APPLY_HIERARCHY_AFTER_UPCODING)
}

scenario_manifest <- bind_rows(manifest_rows)
write_csv(scenario_manifest, file.path(OUT_ROOT, "scenario_manifest.csv"))

message("Done. Export folder: ", normalizePath(OUT_ROOT))
