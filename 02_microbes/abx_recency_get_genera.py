import pandas as pd
import numpy as np

# Read and sort abx use data
abx_covar = pd.read_csv("../data/abx_covariates_with_statins_a1c_all_covars.csv")
abx_covar = abx_covar.sort_values(by="days_in_program")

# Read and sort ASVS
asvs = pd.read_csv("/proj/gibbons/arivale_16S/data/asvs.csv")
metadata = pd.read_csv("/proj/gibbons/arivale_16S/data/metadata.csv")
asvs_meta = pd.merge(asvs, metadata[["id", "public_client_id", "sex", "age", "days_in_program"]], on="id")

# Get taxonomy information
tax = pd.read_csv("/proj/gibbons/arivale_16S/data/taxonomy.csv")
tax = tax.drop("sequence", axis=1).rename(columns={"id": "hash"})
asvs_meta_tax = pd.merge(asvs_meta, tax, on="hash")

# Check that individuals only show up once (no more than 1 day_in_program observation)
# Drops rows that have the same values in both col
appearances = asvs_meta_tax[[
    'public_client_id','days_in_program'
]].drop_duplicates().sort_values(
    by=['public_client_id','days_in_program'])  

# List of people who show up more than once
duplicate_ids = set(
    appearances[appearances.public_client_id.duplicated()]['public_client_id'].tolist()
)

# Smaller df to see which days they show up
duplicate_rows = appearances[appearances.public_client_id.isin(duplicate_ids)].sort_values(
    by=['public_client_id','days_in_program'])

# Select the specific days_in_program we want
abx_days = abx_covar[[
    'public_client_id', 'days_in_program', 'abx_recency'
]].copy()

abx_days = abx_days.sort_values(by='days_in_program')

asvs_meta_tax = asvs_meta_tax.dropna(
    subset=['public_client_id', 'days_in_program']
).sort_values(by='days_in_program')

asvs_abx = pd.merge_asof(
    asvs_meta_tax, abx_days,
    by="public_client_id",
    on="days_in_program",
    direction="nearest", 
    tolerance=30.0).dropna(subset=['id','abx_recency']) # Keep only folks with stool and abx questionnaire data

asvs_abx = asvs_abx[asvs_abx['id'] != "AV15-3725|AKE025"] 

taxa = asvs_abx.drop(columns=["hash","abx_recency"], axis=1)

taxa = taxa.assign(taxa_genus_level = taxa.Kingdom.astype(str) + '_' + \
                   taxa.Phylum.astype(str) + '_' + taxa.Class.astype(str) + '_' + \
                   taxa.Order.astype(str) + '_' + taxa.Family.astype(str) + '_' + \
                   taxa.Genus.astype(str))

genera = taxa.groupby(["id", "taxa_genus_level"])["count"].sum().reset_index()

# Count how many times the string 'nan' appears in each row
# In particular there are Eukaryota that show up with many nans
too_many_nans_genus = taxa['taxa_genus_level'].str.count('_nan') > 3

# Drop those rows
genus_more_specific = taxa[~too_many_nans_genus].sort_values(by = ['public_client_id'])

save_taxonomy = genus_more_specific[[
    'Kingdom','Phylum','Class','Order','Family','Genus','taxa_genus_level'
]].drop_duplicates()

save_taxonomy.to_csv("../data/tax_genus_level.csv", index=False)

arrays = [genus_more_specific['public_client_id'], genus_more_specific['taxa_genus_level']]
index = pd.MultiIndex.from_arrays(arrays, names=('public_client_id','taxa_genus_level'))

# Pivot df with taxa as columns, public_client_id as index, and counts as values
taxa_melt_genus = pd.DataFrame(data = {'count': genus_more_specific['count'].to_numpy()}, index=index)

taxa_pivot_genus = taxa_melt_genus.pivot_table(index='public_client_id', columns='taxa_genus_level', 
                                         values='count', aggfunc='sum', fill_value = 0)

taxa_pivot_genus = taxa_pivot_genus.dropna(how='any')

taxa_pivot_genus.drop(columns=[
    'Eukaryota_Euglenozoa_Euglenida_Aphagea_Aphagea_fa_Distigma',
    'Eukaryota_Euglenozoa_Euglenida_Aphagea_Aphagea_fa_nan',
    'Eukaryota_Euglenozoa_Euglenida_nan_nan_nan',
    'Eukaryota_Incertae_Sedis_Incertae_Sedis_cl_Incertae_Sedis_or_Incertae_Sedis_fa_Blastocystis',
    'Eukaryota_Parabasalia_Tritrichomonadea_Tritrichomonadea_or_Tritrichomonadea_fa_Dientamoeba'],
                      inplace=True)

taxa_pivot_genus.reset_index(inplace=True)
taxa_pivot_genus.columns.name = None

###### Save raw count table #####
taxa_pivot_genus.to_csv("../data/rh_abx_genus_level_counts.csv", index=False)