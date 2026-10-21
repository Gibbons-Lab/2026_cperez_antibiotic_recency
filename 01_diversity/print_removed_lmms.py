import pandas as pd
import arivale_data_interface as adi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# apply settings so the lib knows where the data is
adi.apply_isb_config()

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
# Now you can import your file
from regression_functions import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

abx_covar = pd.read_csv("../data/rh_abx_covar_alphadivs_scfa.csv", dtype={'public_client_id':str})

# Use snapshot to identify repeat clients
div = adi.get_snapshot("microbiome_diversity")
div.dropna(subset=['days_in_program'], inplace=True)
shannon_div = div[[
    "diversity_shannon", "public_client_id",
      "days_in_program", "vendor_dashboard", "observed_otus_20000"]].copy()

shannon_div = shannon_div[shannon_div.public_client_id.isin(
    abx_covar.public_client_id)].sort_values(by='public_client_id')

# Grab the ids that show up more than once
shannon_dup_id = (shannon_div[shannon_div.public_client_id.duplicated()
                  ].public_client_id.tolist())

shannon_dup = shannon_div[shannon_div.public_client_id.isin(shannon_dup_id)]

shannon_dup['public_client_id'].nunique() # Sizeable number!

# Now subset these folks from our study
abx_covar_repeat = abx_covar[abx_covar.public_client_id.isin(shannon_dup_id)]
abx_covar_repeat_small = (abx_covar_repeat[[
    'public_client_id', 
    'days_in_program',
    'abx_recency']].copy())

# Want to be able to compare the dates
abx_covar_repeat_small.rename(columns={
    'days_in_program':'days_in_program_questionnaire'}, inplace=True)


shannon_repeats = shannon_dup.merge(
    abx_covar_repeat_small,
    how ='left',
    on="public_client_id")

# Check if stool is close to abx questionnaire
# Aka assign baseline and follow up
shannon_repeats['paired_stool_questionnare'] = (
    (shannon_repeats['days_in_program'] - shannon_repeats['days_in_program_questionnaire'] <= 30)
    .map({True: 'Yes', False: 'No'}))

shannon_repeats_baseline = shannon_repeats[shannon_repeats['paired_stool_questionnare'] == 'Yes'].drop_duplicates(
    subset='public_client_id')

shannon_repeats_followup = shannon_repeats[shannon_repeats['paired_stool_questionnare'] == 'No']

shannon_repeats_followup.sort_values(by=['public_client_id', 'days_in_program'])
shannon_repeats_followup_first = shannon_repeats_followup.drop_duplicates(subset='public_client_id', keep='first')

#  Drop this client 
# Samples at day 2 and 25
# To close to one another

to_drop = list(set(shannon_repeats_baseline.public_client_id) - 
               set(shannon_repeats_followup_first.public_client_id))

shannon_repeats_baseline = (shannon_repeats_baseline[
    shannon_repeats_baseline.public_client_id != to_drop[0]])

all_followup_timepoints = pd.concat(
    [shannon_repeats_baseline, shannon_repeats_followup_first]
    )

abx_df_obs_feats = (abx_covar[[
    'public_client_id',
    'days_in_program', 
    'observed_features', 
    'berger_parker',
    'shannon_index']].copy())

abx_df_obs_feats.rename(columns={
    'days_in_program':'days_in_program_abx_df'}, inplace=True)

baseline_merge = (shannon_repeats_baseline[[
    'public_client_id',
    'days_in_program',
    'days_in_program_questionnaire',
    'paired_stool_questionnare',
    'abx_recency']].copy())

baseline_alphas = baseline_merge.merge(
    abx_df_obs_feats, 
    how='left',
    on='public_client_id')
baseline_alphas.head()

baseline_alphas.drop(columns=['days_in_program_abx_df'], inplace=True)

follow_up_small = (shannon_repeats_followup_first[[
    'public_client_id',
    'days_in_program',
    'paired_stool_questionnare',
    'abx_recency']].copy())

# Find follow up samples
asvs = pd.read_csv("/proj/gibbons/arivale_16S/data/asvs.csv")
metadata = pd.read_csv("/proj/gibbons/arivale_16S/data/metadata.csv")
asvs_meta = pd.merge(asvs, metadata[["id", "public_client_id", "sex", "age", "days_in_program"]], on="id")
tax = pd.read_csv("/proj/gibbons/arivale_16S/data/taxonomy.csv")
tax = tax.drop("sequence", axis=1).rename(columns={"id": "hash"})
asvs_meta_tax = pd.merge(asvs_meta, tax, on="hash")

# We have to subset down to only the id/date combinations we're interested in
follow_up_small = follow_up_small.sort_values(by='days_in_program')
asvs_meta_tax = asvs_meta_tax.dropna(
    subset=['public_client_id', 'days_in_program']
).sort_values(by='days_in_program')

asvs_abx = pd.merge_asof(
    asvs_meta_tax, follow_up_small,
    by="public_client_id",
    on="days_in_program",
    direction="nearest", 
    tolerance=30.0).dropna(                     # Keep only folks with 
    subset=['id', 'paired_stool_questionnare']) # stool and abx questionnaire data

taxa = asvs_abx.drop(columns=["hash"], axis=1)

# Make a new column, similar to before, but exclude the species in the name
taxa = taxa.assign(taxa_genus_level = taxa.Kingdom.astype(str) + '_' + \
                   taxa.Phylum.astype(str) + '_' + taxa.Class.astype(str) + '_' + \
                   taxa.Order.astype(str) + '_' + taxa.Family.astype(str) + '_' + \
                   taxa.Genus.astype(str))

genera = taxa.groupby(["id", "taxa_genus_level"])["count"].sum().reset_index()
genera.taxa_genus_level.nunique()

too_many_nans_genus = taxa['taxa_genus_level'].str.count('_nan') > 3
genus_more_specific = taxa[~too_many_nans_genus].sort_values(by = ['public_client_id'])

print(f"minimum days in program for follow up: {genus_more_specific.days_in_program.min()}")

# Create MultiIndex for pivot table
arrays = [genus_more_specific['public_client_id'], genus_more_specific['taxa_genus_level']]
index = pd.MultiIndex.from_arrays(arrays, names=('public_client_id','taxa_genus_level'))

# Pivot df with taxa as columns, public_client_id as index, and counts as values
taxa_melt_genus = pd.DataFrame(data = {'count': genus_more_specific['count'].to_numpy()}, index=index)

taxa_pivot_genus = taxa_melt_genus.pivot_table(index='public_client_id', columns='taxa_genus_level', 
                                         values='count', aggfunc='sum', fill_value = 0)

taxa_pivot_genus = taxa_pivot_genus.dropna(how='any')
taxa_pivot_genus.reset_index(inplace=True)
taxa_pivot_genus.columns.name = None
eukaryota_cols = taxa_pivot_genus.columns[taxa_pivot_genus.columns.str.contains('Eukaryota')]
taxa_pivot_genus.drop(columns=eukaryota_cols, inplace=True)

# Remove folks known to use many abx
multiple_use = pd.read_csv('../data/abx_multiple_use_df.csv', dtype={'public_client_id':str})

clean_follow_up_df = (taxa_pivot_genus[~taxa_pivot_genus.public_client_id.isin(
    multiple_use.public_client_id)])

total_dropped = len(to_drop) + (len(taxa_pivot_genus) - len(clean_follow_up_df))
print(f'total dropped so far: {total_dropped}')

clean_follow_up_df.to_csv("../data/repeat_client_follow_up_genera_counts.csv", index=False)

### AFTER RAREFICATION ###
followup_alphas = pd.read_csv("../data/abx_followups_alphadivs_rarefied.csv", dtype={'public_client_id':str})
followup_alphas = follow_up_small.merge(followup_alphas, how='left', on='public_client_id')
print(f'unique clients: {followup_alphas.public_client_id.nunique()}')
followup_alphas_long = followup_alphas.copy()

# Now going to try and make long_df to use in a lmm
followup_alphas_long.rename(columns={'observed_features_followup':'observed_features'}, inplace=True)
followup_alphas_long.rename(columns={'shannon_index_followup':'shannon_index'}, inplace=True)
followup_alphas_long.rename(columns={'berger_parker_followup':'berger_parker'}, inplace=True)

# First let's drop the helper paired stool column from both questionnaires
baseline_alphas.drop(columns='paired_stool_questionnare', inplace=True)
followup_alphas_long.drop(columns='paired_stool_questionnare', inplace=True)
baseline_alphas.drop(columns='days_in_program_questionnaire', inplace=True)

followup_alphas.drop(columns=['paired_stool_questionnare','abx_recency'], inplace=True)
followup_alphas.rename(columns={'days_in_program':'days_in_program_followup'}, inplace=True)

followup_baselines_all = baseline_alphas.merge(followup_alphas, how='left', on='public_client_id') # used in wide df

baseline_alphas_long = baseline_alphas.copy()
abx_covar_followups = abx_covar[['public_client_id', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                 'meds_blood_sugar', 'meds_cholesterol', 'sex', 'age',
                                 'centered_eGFR','abx_recency_three', 'abx_recency_twelve']].copy()

abx_covar_followups = abx_covar_followups[abx_covar_followups.public_client_id.isin(baseline_alphas_long.public_client_id)]

baseline_alphas_long = baseline_alphas_long.merge(abx_covar_followups, how='left', on='public_client_id')
followup_alphas_long = followup_alphas_long.merge(abx_covar_followups, how='left', on='public_client_id')

# Make a days_since_baseline column
baseline_alphas_long.set_index('public_client_id', inplace=True)
followup_alphas_long.set_index('public_client_id', inplace=True)

baseline_alphas_long.loc[:,'dsb'] = 0
followup_alphas_long.loc[:,'dsb'] = followup_alphas_long.loc[:,'days_in_program'] - baseline_alphas_long.loc[:,'days_in_program']
fu_long_df = pd.concat([baseline_alphas_long, followup_alphas_long], axis=0)

vendor_df = shannon_repeats[['public_client_id','days_in_program','vendor_dashboard']].copy()

vendor_df = vendor_df.sort_values(by='days_in_program')
fu_long_df = fu_long_df.sort_values(by='days_in_program')
long_df = pd.merge_asof(
    fu_long_df, 
    vendor_df,
    on='days_in_program',
    by='public_client_id').sort_values(
    by=['public_client_id','days_in_program'])

print(f'unique clients in long df: {long_df.public_client_id.nunique()}')

drop_na_vendor = long_df[long_df.vendor_dashboard.isna()].public_client_id.tolist()
print(f'dropped for lack of vendor at follow up: {len(drop_na_vendor)}')

long_df = long_df[~long_df.public_client_id.isin(drop_na_vendor)]

weights = adi.get_snapshot("weight")[[
    "BMI_CALC",
    "public_client_id",
    "days_in_program"]].dropna()

weights = weights.sort_values(by='days_in_program')
long_df = long_df.sort_values(by='days_in_program')

long_bmi = pd.merge_asof(
    long_df, 
    weights,
    on='days_in_program',
    by='public_client_id')

drop_na_bmi = long_bmi[long_bmi.BMI_CALC.isna()].public_client_id.tolist()
print(f'dropped for lack of BMI at follow up: {len(drop_na_bmi)}')
long_bmi = long_bmi[~long_bmi.public_client_id.isin(drop_na_bmi)]

sus_fu_ids = long_bmi[long_bmi.observed_features < 35].public_client_id.tolist()
sus_fu_ids_dip = long_bmi[long_bmi.observed_features < 35][['public_client_id','days_in_program']].copy()
print(f'dropped for unknown low reads: {len(sus_fu_ids)}')

long_bmi = long_bmi[~long_bmi.public_client_id.isin(sus_fu_ids)]

print(long_bmi.public_client_id.nunique())

print(set(long_bmi.public_client_id.unique()) - set(all_followup_timepoints.public_client_id.unique()))
