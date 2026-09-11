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

# Specifying colClass so leading 0 is not dropped
# Using this one bc I added stool vendor already
metadata = read.csv("../data/abx_covariates_with_statins_a1c_all_covars.csv",
                    colClasses = c(public_client_id = "character"))

filtered_metadata = filter(metadata, public_client_id %in% counts$public_client_id)
filtered_metadata$meds_blood_sugar <- factor(filtered_metadata$meds_blood_sugar)
filtered_metadata$meds_cholesterol <- factor(filtered_metadata$meds_cholesterol)
filtered_metadata$vendor_dashboard <- factor(filtered_metadata$vendor_dashboard)
filtered_metadata$abx_recency <- factor(filtered_metadata$abx_recency)

##### ------- Create matrix
# Name is the variable for ASVs
# Value is their count
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

##### ------- Calculate distance
# Setting a seed for reproducibility
set.seed(3) 
dist_bray <- vegdist(counts_matrix, dmethod = "bray", binary=FALSE)
dist_jaccard <- vegdist(counts_matrix, dmethod = "jaccard", binary=TRUE)

# Save an RDS file to skip distance calc in future
saveRDS(dist_bray, "../data/rh_distance_matrix_bray.rds")
saveRDS(dist_jaccard, "../data/rh_distance_matrix_jaccard.rds")

##### ------- Calculate distance
pcoa_result_bray <- cmdscale(dist_bray, k = 5, eig = TRUE, add=TRUE)
pcoa_result_jaccard <- cmdscale(dist_jaccard, k = 5, eig = TRUE, add=TRUE)

saveRDS(pcoa_result_bray, "../data/abx_recency_pcoa_bray.rds")
saveRDS(pcoa_result_jaccard, "../data/abx_recency_pcoa_jaccard.rds")

groups = factor(filtered_metadata$abx_recency,
                levels = c("none", "twelve", "three"))

##### ------- Save coordinates to CSV file: BRAY
coordinates_bray <- as.matrix(pcoa_result_bray$points)

# Use lapply to create lists of coordinates
pco_vectors_bray <- lapply(1:5, function(i) coordinates_bray[, i])

# Convert the list to a dataframe and set the column names
pcoa_df_bray <- setNames(data.frame(pco_vectors_bray), paste0("pcoa", 1:5))

pcoa_df_bray$public_client_id <- rownames(coordinates_bray)
pcoa_df_bray <- left_join(pcoa_df_bray,
                          select(filtered_metadata, public_client_id, abx_recency),
                          by = "public_client_id")

# Save this df as a CSV to run in python regressions
write.csv(pcoa_df_bray, "../data/abx_recency_pcoa_df_bray.csv", row.names = FALSE)

##### ------- Save coordinates to CSV file: Jaccard
coordinates_jaccard <- as.matrix(pcoa_result_jaccard$points)

# Use lapply to create lists of coordinates
pco_vectors_jaccard <- lapply(1:5, function(i) coordinates_jaccard[, i])

# Convert the list to a dataframe and set the column names
pcoa_df_jaccard <- setNames(data.frame(pco_vectors_jaccard), paste0("pcoa", 1:5))

pcoa_df_jaccard$public_client_id <- rownames(coordinates_jaccard)
pcoa_df_jaccard <- left_join(pcoa_df_jaccard,
                          select(filtered_metadata, public_client_id, abx_recency),
                          by = "public_client_id")

# Save this df as a CSV to run in python regressions
write.csv(pcoa_df_jaccard, "../data/abx_recency_pcoa_df_jaccard.csv", row.names = FALSE)

##### ------- Generate dfs for PERMANOVA
merged_metadata_bray <- merge(pcoa_df_bray,
                              select(filtered_metadata, -abx_recency),
                              by = "public_client_id")

merged_metadata_bray <- merged_metadata_bray[
  match(rownames(counts_matrix), merged_metadata_bray$public_client_id), ]
stopifnot(identical(rownames(counts_matrix), merged_metadata_bray$public_client_id))

merged_metadata_jaccard <- merge(pcoa_df_jaccard,
                                 select(filtered_metadata, -abx_recency),
                                 by = "public_client_id")

merged_metadata_jaccard <- merged_metadata_jaccard[
  match(rownames(counts_matrix), merged_metadata_jaccard$public_client_id), ]
stopifnot(identical(rownames(counts_matrix), merged_metadata_jaccard$public_client_id))

##### ------- PERMANOVA: Bray
set.seed(3)
permanova_result_bray <- adonis2(dist_bray ~  age + sex + centered_BMI + 
                              meds_blood_sugar + meds_cholesterol + vendor_dashboard +
                              PC1 + PC2 + PC3 + PC4 + PC5 + centered_eGFR + abx_recency, 
                            data = merged_metadata_bray, 
                            by = "margin",
                            permutations = 9999)

saveRDS(permanova_result_bray, "../data/beta_div_PERMANOVA_bray.rds")

##### ------- PERMANOVA: Jaccard
set.seed(3)
permanova_result_jaccard <- adonis2(dist_jaccard ~  age + sex + centered_BMI + 
                              meds_blood_sugar + meds_cholesterol + vendor_dashboard +
                              PC1 + PC2 + PC3 + PC4 + PC5 + centered_eGFR + abx_recency, 
                            data = merged_metadata_jaccard, 
                            by = "margin",
                            permutations = 9999)

saveRDS(permanova_result_jaccard, "../data/beta_div_PERMANOVA_jaccard.rds")
