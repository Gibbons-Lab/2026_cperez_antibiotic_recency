#!/usr/bin/env Rscript

# Load libs
library(permute)
library(vegan)
library(ecodist)
library(glue)
library(dplyr)
library(tidyr)

##### ------- Read in data
counts = read.csv("../data/rh_abx_cohort_rarefied_counts.csv",
                  colClasses = c(X = "character"))
colnames(counts)[1] <- "public_client_id"

counts$days_in_program <- NULL

counts = counts %>% pivot_longer(-public_client_id) %>%  # Rows are person, ASVs count
  group_by(public_client_id) %>%                         # Groups by person
  group_by(name) %>%                                     # Group by taxa  
  mutate(total = sum(value)) %>%                         # Sums ASVS over all folks
  filter(total != 0) %>%                                 # Keep taxa that not 0
  ungroup() %>%
  pivot_wider(id_cols = public_client_id) %>%
  as.data.frame()

# Add clients as row names, drop that id column
rownames(counts) <- counts$public_client_id
counts$public_client_id <- NULL

# Need to have it in matrix form for vegan
counts_matrix <- as.matrix(counts)

dist_jaccard <- readRDS("../data/rh_distance_matrix_jaccard.rds")
dist_bray <- readRDS("../data/rh_distance_matrix_bray.rds")

metadata = read.csv("../data/abx_covariates_with_statins_a1c_all_covars.csv",
                    colClasses = c(public_client_id = "character"))

metadata$meds_blood_sugar <- factor(metadata$meds_blood_sugar)
metadata$meds_cholesterol <- factor(metadata$meds_cholesterol)
metadata$vendor_dashboard <- factor(metadata$vendor_dashboard)
metadata$abx_recency <- factor(metadata$abx_recency)

pcoa_df_bray <- read.csv("../data/abx_recency_pcoa_df_bray.csv", 
                         colClasses = c(public_client_id = "character"))
pcoa_df_jaccard <- read.csv("../data/abx_recency_pcoa_df_jaccard.csv", 
                            colClasses = c(public_client_id = "character"))

pcoa_df_bray$abx_recency <- factor(pcoa_df_bray$abx_recency, levels = c("none", "twelve", "three"))
pcoa_df_jaccard$abx_recency <- factor(pcoa_df_jaccard$abx_recency, levels = c("none", "twelve", "three"))


##### ------- Generate dfs for PERMANOVA
merged_metadata_bray <- merge(pcoa_df_bray,
                              select(metadata, -abx_recency),
                              by = "public_client_id")

merged_metadata_bray <- merged_metadata_bray[
  match(rownames(counts_matrix), merged_metadata_bray$public_client_id), ]
stopifnot(identical(rownames(counts_matrix), merged_metadata_bray$public_client_id))

merged_metadata_jaccard <- merge(pcoa_df_jaccard,
                                 select(metadata, -abx_recency),
                                 by = "public_client_id")

merged_metadata_jaccard <- merged_metadata_jaccard[
  match(rownames(counts_matrix), merged_metadata_jaccard$public_client_id), ]
stopifnot(identical(rownames(counts_matrix), merged_metadata_jaccard$public_client_id))

model_formula <- ~ age + sex + centered_BMI + meds_blood_sugar + meds_cholesterol +
  vendor_dashboard + PC1 + PC2 + PC3 + PC4 + PC5 + centered_eGFR + abx_recency

##### ------- Run pairwise PERMANOVA
pairwise_adonis <- function(dist_matrix, metadata, group_var = "abx_recency",
                            formula, permutations = 9999,
                            id_var = "public_client_id") {
  
  dm <- as.matrix(dist_matrix)          
  groups <- levels(droplevels(factor(metadata[[group_var]])))
  results <- list()
  k <- 1
  
  for (i in 1:(length(groups) - 1)) {
    for (j in (i + 1):length(groups)) {
      
      g1 <- groups[i]
      g2 <- groups[j]
      
      subset_meta <- metadata[metadata[[group_var]] %in% c(g1, g2), ]
      subset_meta <- droplevels(subset_meta)         
      
      ids <- as.character(subset_meta[[id_var]])      
      stopifnot(all(ids %in% rownames(dm)))           
      
      subset_dist <- as.dist(dm[ids, ids])
      
      model  <- update(formula, subset_dist ~ .)
      result <- adonis2(model, data = subset_meta,
                        permutations = permutations, by = "margin")
      
      results[[k]] <- data.frame(
        group1   = g1,
        group2   = g2,
        p_value  = result[group_var, "Pr(>F)"],
        R2       = result[group_var, "R2"],
        pseudo_F = result[group_var, "F"]
      )
      k <- k + 1
    }
  }
  
  pairwise_results <- do.call(rbind, results)
  pairwise_results$padj <- p.adjust(pairwise_results$p_value, method = "BH")
  rownames(pairwise_results) <- NULL
  pairwise_results
}

###### ------- Run pairwise betadisper
pairwise_betadisper <- function(dist_matrix, metadata, group_var = "abx_recency",
                                permutations = 9999, id_var = "public_client_id") {
  
  dm <- as.matrix(dist_matrix)
  groups <- levels(droplevels(factor(metadata[[group_var]])))
  results <- list()
  k <- 1
  
  for (i in 1:(length(groups) - 1)) {
    for (j in (i + 1):length(groups)) {
      
      g1 <- groups[i]; g2 <- groups[j]
      
      subset_meta <- droplevels(metadata[metadata[[group_var]] %in% c(g1, g2), ])
      ids <- as.character(subset_meta[[id_var]])
      stopifnot(all(ids %in% rownames(dm)))
      
      subset_dist <- as.dist(dm[ids, ids])
      grp <- factor(subset_meta[[group_var]])
      
      bd <- betadisper(subset_dist, grp)          
      pt <- permutest(bd, permutations = permutations)
      
      results[[k]] <- data.frame(
        group1       = g1,
        group2       = g2,
        disp_g1      = tapply(bd$distances, grp, mean)[g1],  # mean dist-to-median
        disp_g2      = tapply(bd$distances, grp, mean)[g2],
        disp_F       = pt$tab[1, "F"],
        disp_p_value = pt$tab[1, "Pr(>F)"]
      )
      k <- k + 1
    }
  }
  
  out <- do.call(rbind, results)
  out$disp_padj <- p.adjust(out$disp_p_value, method = "BH")  # kept separate from
  rownames(out) <- NULL                                       # PERMANOVA family.
  out
}

# =============== 3. Global dispersion (companion to global PERMANOVA) =========
global_betadisper <- function(dist_matrix, metadata, group_var = "abx_recency",
                              permutations = 9999, id_var = "public_client_id") {
  dm  <- as.matrix(dist_matrix)
  ids <- as.character(metadata[[id_var]])
  stopifnot(all(ids %in% rownames(dm)))
  
  d   <- as.dist(dm[ids, ids])
  grp <- factor(metadata[[group_var]])
  bd  <- betadisper(d, grp)
  list(betadisper = bd,
       permutest  = permutest(bd, permutations = permutations),
       tukeyHSD   = TukeyHSD(bd))   # pairwise median-distance contrasts.
}

# ============================= 4. Driver / runner =============================
# --- PERMANOVA (median/location) ---
jaccard_results <- pairwise_adonis(dist_jaccard, merged_metadata_jaccard,
                                   formula = model_formula)
saveRDS(jaccard_results, "../data/jaccard_PERMANOVA_result_pairwise.rds") 

bray_results <- pairwise_adonis(dist_bray, merged_metadata_bray,           
                                formula = model_formula)                  
saveRDS(bray_results, "../data/bray_PERMANOVA_result_pairwise.rds")

# --- Dispersion (betadisper) diagnostic ---
jaccard_disp <- pairwise_betadisper(dist_jaccard, merged_metadata_jaccard)
saveRDS(jaccard_disp, "../data/jaccard_betadisper_pairwise.rds")

bray_disp <- pairwise_betadisper(dist_bray, merged_metadata_bray)
saveRDS(bray_disp, "../data/bray_betadisper_pairwise.rds")

# --- Global dispersion (optional companion) ---
jaccard_disp_global <- global_betadisper(dist_jaccard, merged_metadata_jaccard)
bray_disp_global    <- global_betadisper(dist_bray,    merged_metadata_bray)
saveRDS(jaccard_disp_global, "../data/jaccard_betadisper_global.rds")
saveRDS(bray_disp_global,    "../data/bray_betadisper_global.rds")
