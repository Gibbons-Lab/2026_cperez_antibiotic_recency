#!/usr/bin/env Rscript
# Dispersion-matched PERMANOVA sensitivity analysis
# Date: Aug 17, 2026

# Load libs
library(vegan)
library(dplyr)

## ---- 0a. Input distance matrices 
# >>> CHANGE ME: paths to your saved dist objects (one per metric).
DIST_PATHS <- list(
  jaccard = "../data/rh_distance_matrix_jaccard.rds",
  bray    = "../data/rh_distance_matrix_bray.rds"
)

## ---- 0b. Metadata 
METADATA_PATH <- "../data/abx_covariates_with_statins_a1c_all_covars.csv"

ID_VAR <- "public_client_id"

GROUP_VAR    <- "abx_recency"
GROUP_LEVELS <- c("none", "twelve", "three")

COVARIATE_FORMULA <- ~ age + sex + centered_BMI + meds_blood_sugar +
  meds_cholesterol + vendor_dashboard + PC1 + PC2 + PC3 + PC4 + PC5 +
  centered_eGFR

## ---- 0c. Prior results  
FULL_RESULT_PATHS <- list(
  jaccard = list(
    global_permanova   = "../data/beta_div_PERMANOVA_jaccard.rds",
    pairwise_permanova = "../data/jaccard_PERMANOVA_result_pairwise.rds",
    betadisper_global  = "../data/jaccard_betadisper_global.rds",
    betadisper_pw      = "../data/jaccard_betadisper_pairwise.rds"
  ),
  bray = list(
    global_permanova   = "../data/beta_div_PERMANOVA_bray.rds",
    pairwise_permanova = "../data/bray_PERMANOVA_result_pairwise.rds",
    betadisper_global  = "../data/bray_betadisper_global.rds",
    betadisper_pw      = "../data/bray_betadisper_pairwise.rds"
  )
)

## ---- 0d. Analysis params
MATCH_P_THRESHOLD <- 0.10   # stop trimming once permutest p > this (dispersion "matched")
MAX_DROP_FRACTION <- 0.30   # cap: never drop more than this fraction of the more-dispersed group
DROP_STEP         <- 1      # how many points to drop per iteration (1 = finest, slower)
CHECK_EVERY       <- 2      # run permutest to test the stop rule every this-many drops (speed knob)
PERMS_STOP        <- 999    # permutations for the cheap in-loop stopping test
PERMS_FINAL       <- 9999   # permutations for the final reported permutest / adonis2
N_SEEDS           <- 100    # randomized pass: number of matched subsets to draw
BASE_SEED         <- 1      # base seed; randomized draws use BASE_SEED + 1..N_SEEDS
TRIM_WEIGHT_POWER <- 6      # randomized pass: drop-probability ∝ dist_to_median^POWER.
#   large  -> near-deterministic (tail dropped first)
#   0      -> uniform random subsampling
BETADISPER_TYPE   <- "median"  # vegan default (Anderson 2006 robust choice); pinned explicitly.
OUTPUT_CSV        <- "dispersion_matched_permanova_summary.csv"

# =============================================================================
# 1. LOAD DATA
# =============================================================================
message("Loading metadata: ", METADATA_PATH)
metadata <- read.csv(METADATA_PATH, colClasses = setNames("character", ID_VAR))
metadata[[GROUP_VAR]] <- factor(metadata[[GROUP_VAR]], levels = GROUP_LEVELS)

dists <- lapply(DIST_PATHS, function(p) {
  message("Loading distance object: ", p)
  d <- readRDS(p)
  stopifnot(inherits(d, "dist"))   # must be a dist object, not a matrix
  d
})

# =============================================================================
# 2. HELPERS
# =============================================================================

# Subset a dist object to a set of IDs (by name), returning a dist.
# Pattern from the lab's pairwise_adonis: as.matrix ONCE, subset by name, as.dist back.
subset_dist <- function(dist_obj, ids) {
  m <- as.matrix(dist_obj)
  stopifnot(all(ids %in% rownames(m)))
  as.dist(m[ids, ids, drop = FALSE])
}

# Align metadata rows to a set of IDs and drop unused factor levels.
subset_meta <- function(meta, ids) {
  m <- meta[match(ids, meta[[ID_VAR]]), , drop = FALSE]
  m[[GROUP_VAR]] <- droplevels(m[[GROUP_VAR]])
  m
}

# betadisper distance-to-median for a 2-group subset; returns list(bd, perm_p, means).
disp_diag <- function(dist_obj, meta, ids, perms) {
  d   <- subset_dist(dist_obj, ids)
  m   <- subset_meta(meta, ids)
  grp <- m[[GROUP_VAR]]
  bd  <- betadisper(d, grp, type = BETADISPER_TYPE)
  pt  <- permutest(bd, permutations = perms)
  list(
    bd     = bd,
    perm_p = pt$tab[1, "Pr(>F)"],
    means  = tapply(bd$distances, grp, mean)
  )
}

# adonis2 with the prespecified covariates + abx_recency on an ID subset.
# Returns the abx_recency row (R2, F, p) plus n.
adonis_on_subset <- function(dist_obj, meta, ids, perms) {
  d <- subset_dist(dist_obj, ids)
  m <- subset_meta(meta, ids)
  # Build full formula: dist ~ covariates + abx_recency  (marginal effects).
  full_formula <- update(COVARIATE_FORMULA,
                         paste("d ~ . +", GROUP_VAR))
  res <- adonis2(full_formula, data = m, permutations = perms, by = "margin")
  data.frame(
    n        = length(ids),
    abx_R2   = res[GROUP_VAR, "R2"],
    abx_F    = res[GROUP_VAR, "F"],
    abx_p    = res[GROUP_VAR, "Pr(>F)"]
  )
}

# Identify the more-dispersed group of a pair from a betadisper object.
more_dispersed_group <- function(means) names(means)[which.max(means)]

# -----------------------------------------------------------------------------
# Core trimmer. Removes points from the MORE-DISPERSED group until permutest
# p > MATCH_P_THRESHOLD or the drop cap is hit.
#   mode = "deterministic": drop the current farthest-from-median point(s).
#   mode = "random": sample which point to drop, weighted by dist^TRIM_WEIGHT_POWER
#                    (seed-dependent; call set.seed() before this).
# Returns the surviving ids + bookkeeping.
# -----------------------------------------------------------------------------
trim_to_match <- function(dist_obj, meta, ids_pair, mode = c("deterministic", "random")) {
  mode <- match.arg(mode)
  ids  <- ids_pair
  
  # Initial dispersion diagnostic (cheap perms) to pick the more-dispersed group.
  dd0    <- disp_diag(dist_obj, meta, ids, PERMS_STOP)
  hi_grp <- more_dispersed_group(dd0$means)
  
  # Degenerate case: permutest returns NA/NaN (e.g. zero within-group dispersion,
  # or identical distances). Nothing to trim; report as-is so the pair doesn't
  # crash the run. This should not happen on real Bray/Jaccard microbiome data.
  if (!is.finite(dd0$perm_p)) {
    dd_final <- disp_diag(dist_obj, meta, ids, PERMS_FINAL)
    return(list(ids = ids, hi_group = hi_grp,
                n_hi_start = sum(subset_meta(meta, ids)[[GROUP_VAR]] == hi_grp),
                n_dropped = 0L, frac_dropped = 0,
                matched = isTRUE(dd_final$perm_p > MATCH_P_THRESHOLD),
                disp_p_final = dd_final$perm_p, cap_hit = FALSE))
  }
  
  meta_pair <- subset_meta(meta, ids)
  hi_ids    <- as.character(meta_pair[[ID_VAR]][meta_pair[[GROUP_VAR]] == hi_grp])
  n_hi_start <- length(hi_ids)
  max_drop   <- floor(MAX_DROP_FRACTION * n_hi_start)
  
  dropped <- 0L
  cur_p   <- dd0$perm_p
  iter    <- 0L
  
  while (cur_p <= MATCH_P_THRESHOLD && dropped < max_drop) {
    # Current distance-to-median for the high-dispersion group's surviving members.
    dd  <- disp_diag(dist_obj, meta, ids, PERMS_STOP)
    d2m <- dd$bd$distances
    grp <- dd$bd$group
    hi_now <- names(d2m)[grp == hi_grp]
    d2m_hi <- d2m[hi_now]
    
    k <- min(DROP_STEP, length(d2m_hi) - 1L, max_drop - dropped)
    if (k <= 0) break
    
    if (mode == "deterministic") {
      drop_ids <- names(sort(d2m_hi, decreasing = TRUE))[seq_len(k)]
    } else {
      w <- (d2m_hi)^TRIM_WEIGHT_POWER          # weight drop prob toward the tail
      w[!is.finite(w)] <- 0
      drop_ids <- sample(names(d2m_hi), size = k, replace = FALSE, prob = w)
    }
    
    ids     <- setdiff(ids, drop_ids)
    dropped <- dropped + k
    iter    <- iter + 1L
    
    # Only pay for a permutest every CHECK_EVERY iterations (speed).
    if (iter %% CHECK_EVERY == 0) {
      cur_p <- disp_diag(dist_obj, meta, ids, PERMS_STOP)$perm_p
    }
  }
  
  # Final, high-precision dispersion check on the trimmed set.
  dd_final <- disp_diag(dist_obj, meta, ids, PERMS_FINAL)
  
  list(
    ids           = ids,
    hi_group      = hi_grp,
    n_hi_start    = n_hi_start,
    n_dropped     = dropped,
    frac_dropped  = dropped / n_hi_start,
    matched       = dd_final$perm_p > MATCH_P_THRESHOLD,
    disp_p_final  = dd_final$perm_p,
    cap_hit       = dropped >= max_drop
  )
}

# =============================================================================
# 3. FULL-DATA REFERENCE VALUES (step 5, read your saved .rds)
# =============================================================================
# Pull the abx_recency row from each saved GLOBAL adonis2, and the saved pairwise
# table, so the trimmed results can be laid beside the numbers you already have.
read_full_refs <- function(metric) {
  paths <- FULL_RESULT_PATHS[[metric]]
  refs  <- list()
  
  # --- global adonis2: extract abx_recency R2/F/p ---
  if (file.exists(paths$global_permanova)) {
    g <- readRDS(paths$global_permanova)
    refs$global <- data.frame(
      metric = metric,
      abx_R2 = g[GROUP_VAR, "R2"],
      abx_F  = g[GROUP_VAR, "F"],
      abx_p  = g[GROUP_VAR, "Pr(>F)"]
    )
  } else {
    warning("Missing global PERMANOVA for ", metric, ": ", paths$global_permanova)
  }
  
  # --- saved pairwise adonis2 table (has p_value, R2, pseudo_F, padj) ---
  if (file.exists(paths$pairwise_permanova)) {
    refs$pairwise <- readRDS(paths$pairwise_permanova)
  } else {
    warning("Missing pairwise PERMANOVA for ", metric, ": ", paths$pairwise_permanova)
  }
  refs
}

# =============================================================================
# 4. MAIN LOOP  --  per metric, per group-pair
# =============================================================================
all_rows <- list()
row_i    <- 1L

for (metric in names(dists)) {
  message("\n================ METRIC: ", metric, " ================")
  dist_obj <- dists[[metric]]
  
  # Keep only metadata rows whose ID is in this distance matrix (and complete covars).
  ids_all  <- rownames(as.matrix(dist_obj))
  meta_m   <- metadata[metadata[[ID_VAR]] %in% ids_all, ]
  meta_m   <- meta_m[!is.na(meta_m[[GROUP_VAR]]), ]
  
  full_refs <- read_full_refs(metric)
  
  grps  <- levels(droplevels(meta_m[[GROUP_VAR]]))
  pairs <- combn(grps, 2, simplify = FALSE)
  
  for (pr in pairs) {
    g1 <- pr[1]; g2 <- pr[2]
    pair_lab <- paste(g1, g2, sep = "_vs_")
    message("---- pair: ", pair_lab, " ----")
    
    ids_pair <- as.character(
      meta_m[[ID_VAR]][meta_m[[GROUP_VAR]] %in% c(g1, g2)]
    )
    
    # ---- (1) betadisper + permutest on the FULL pair (confirm heterogeneity) ----
    dd_full <- disp_diag(dist_obj, meta_m, ids_pair, PERMS_FINAL)
    set.seed(BASE_SEED)  # pin the adonis2 permutation p on full data
    adonis_full <- adonis_on_subset(dist_obj, meta_m, ids_pair, PERMS_FINAL)
    
    # ---- (2+3) DETERMINISTIC trim, then adonis2 on the matched subset ----
    tr_det <- trim_to_match(dist_obj, meta_m, ids_pair, mode = "deterministic")
    set.seed(BASE_SEED)
    adonis_det <- adonis_on_subset(dist_obj, meta_m, tr_det$ids, PERMS_FINAL)
    
    # ---- (4) RANDOMIZED trim over N_SEEDS -> distribution of abx p / R2 ----
    rand_p  <- numeric(N_SEEDS)
    rand_R2 <- numeric(N_SEEDS)
    rand_nd <- integer(N_SEEDS)
    rand_ok <- logical(N_SEEDS)
    for (s in seq_len(N_SEEDS)) {
      set.seed(BASE_SEED + s)
      tr_r <- trim_to_match(dist_obj, meta_m, ids_pair, mode = "random")
      set.seed(BASE_SEED + s)  # pin the adonis2 perms for this draw too
      a_r  <- adonis_on_subset(dist_obj, meta_m, tr_r$ids, PERMS_FINAL)
      rand_p[s]  <- a_r$abx_p
      rand_R2[s] <- a_r$abx_R2
      rand_nd[s] <- tr_r$n_dropped
      rand_ok[s] <- tr_r$matched
    }
    
    # ---- assemble one summary row for this metric x pair ----
    all_rows[[row_i]] <- data.frame(
      metric              = metric,
      pair                = pair_lab,
      more_dispersed_grp  = tr_det$hi_group,
      # full-data dispersion + location
      n_full              = adonis_full$n,
      disp_p_full         = dd_full$perm_p,
      disp_mean_g1_full   = dd_full$means[[g1]],
      disp_mean_g2_full   = dd_full$means[[g2]],
      abx_p_full          = adonis_full$abx_p,
      abx_R2_full         = adonis_full$abx_R2,
      abx_F_full          = adonis_full$abx_F,
      # deterministic matched
      det_n_dropped       = tr_det$n_dropped,
      det_frac_dropped    = round(tr_det$frac_dropped, 3),
      det_cap_hit         = tr_det$cap_hit,
      det_disp_p_matched  = tr_det$disp_p_final,
      det_matched         = tr_det$matched,
      det_n               = adonis_det$n,
      det_abx_p           = adonis_det$abx_p,
      det_abx_R2          = adonis_det$abx_R2,
      det_abx_F           = adonis_det$abx_F,
      # randomized matched (distribution over N_SEEDS)
      rand_n_seeds        = N_SEEDS,
      rand_median_dropped = median(rand_nd),
      rand_frac_matched   = mean(rand_ok),                 # fraction of draws that hit the match target
      rand_abx_p_median   = median(rand_p),
      rand_abx_p_q025     = quantile(rand_p, 0.025),
      rand_abx_p_q975     = quantile(rand_p, 0.975),
      rand_frac_p_lt_05   = mean(rand_p < 0.05),            # <-- key robustness number
      rand_abx_R2_median  = median(rand_R2),
      stringsAsFactors    = FALSE
    )
    row_i <- row_i + 1L
  }
}

summary_tab <- do.call(rbind, all_rows)

# =============================================================================
# 5. REPORT  --  full-data vs dispersion-matched, side by side
# =============================================================================
cat("\n\n===================== DISPERSION-MATCHED PERMANOVA SUMMARY =====================\n")
cat("Interpretation guide:\n")
cat("  * det_matched / rand_frac_matched near 1 -> dispersion successfully equalized.\n")
cat("  * abx SURVIVES (det_abx_p < .05 AND rand_frac_p_lt_05 high) -> strong location shift.\n")
cat("  * abx VANISHES -> AMBIGUOUS (real absence vs. signal-bearing tail was trimmed).\n")
cat("  * Compare det_abx_R2 / rand_abx_R2_median vs abx_R2_full for effect-size shrinkage.\n\n")

print(summary_tab, row.names = FALSE)

write.csv(summary_tab, OUTPUT_CSV, row.names = FALSE)
cat("\nWrote:", OUTPUT_CSV, "\n")

# Optional: echo the saved full-data pairwise tables for direct eyeballing.
for (metric in names(FULL_RESULT_PATHS)) {
  p <- FULL_RESULT_PATHS[[metric]]$pairwise_permanova
  if (file.exists(p)) {
    cat("\n--- saved FULL-data pairwise PERMANOVA (", metric, ") ---\n", sep = "")
    print(readRDS(p))
  }
}

cat("\nDone.\n")
