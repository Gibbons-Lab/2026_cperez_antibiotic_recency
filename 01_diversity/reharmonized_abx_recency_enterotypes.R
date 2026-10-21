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