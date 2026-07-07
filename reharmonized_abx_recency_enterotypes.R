#!usr/bin/env Rscript

## Enterotypes for abx recency cohort
# written June 13 2026

# data wrangling
library(permute)
library(vegan)
library(ggplot2)
library(plyr)
library(dplyr)
library(DirichletMultinomial)

# set wd
setwd("/proj/gibbons/2024_cperez_antibiotics/abx_recency_regressions/reharmonized_abx_cohort")

# data import
counts = read.csv('reharmonized_abx_cohort_rarefied_counts.csv',
                  colClasses = c(X = "character"))

names(counts)[1] <- "public_client_id"
rownames(counts) <- counts$public_client_id
counts$public_client_id <- NULL

counts_matrix <- as.matrix(counts)

fit <- lapply(1:8, dmn, count=counts_matrix, verbose=TRUE,seed=1234)

saveRDS(fit, "harmonized_abx_users_dmm_fit.rds")

# ## Now look at best fit
# best <- which.min(bic)  # or pick manually, e.g. best <- 4
# best_fit <- fit[[best]]
# 
# # Sample-level cluster assignments
# clusters <- apply(mixture(best_fit), 1, which.max)
# result_df <- data.frame(sample = rownames(counts), cluster = clusters)
# 
# saveRDS(result_df, "dmm_clusters_abx_recency.rds", row.names = FALSE)
# 
# # Mixture probabilities per sample
# saveRDS(mixture(best_fit), "dmm_mixture_probs_abx_recency.rds")
# 
# # Feature (taxon) weights per cluster
# feature_weights <- fitted(best_fit)
# saveRDS(feature_weights, "dmm_feature_weights_abx_recency.rds")