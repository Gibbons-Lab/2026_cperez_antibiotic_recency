#!/usr/bin/env Rscript

# Load libs
library(tidyverse)
library(missForest)
library(doParallel)
library(foreach)
registerDoParallel(cores = 5)

# Data
mets_ids_imputed <- read.csv("mets_ids_imputed_test.csv", colClasses = c(public_client_id = "character"))

# Save ids and mets as a matrix
public_client_ids <- mets_ids_imputed$public_client_id
mets_mat <- mets_ids_imputed %>% select(-public_client_id)

set.seed(42)
imputed <- missForest(
  as.data.frame(mets_mat),
  ntree = 100,
  parallelize = 'forests'   # parallelizes across trees
)
mets_ids_imputed <- as.data.frame(imputed$ximp)

mets_ids_imputed <- cbind(public_client_id = public_client_ids, mets_ids_imputed)

write_csv(mets_ids_imputed, 'mets_ids_imputed_missForest.csv')
