import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# apply settings so the lib knows where the data is
import arivale_data_interface as adi
adi.apply_isb_config()

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from regression_functions import *

# Data import
abx_covar = pd.read_csv("../data/rh_abx_covar_alpha_divs_rarefied.csv", dtype={"public_client_id":str})
diet = pd.read_csv("../data/diet_collapsed.csv", dtype={'public_client_id':str})
abx_covar.days_in_program = abx_covar.days_in_program.astype(float)

# Prepare for merge
diet.rename(columns={'vendor':'diet_vendor'}, inplace=True)
diet = diet.sort_values(by="days_in_program").dropna(subset='days_in_program')
abx_covar = abx_covar.sort_values(by="days_in_program")

# Merge with measure
abx_diet = pd.merge_asof(
    abx_covar, diet, 
    by="public_client_id", 
    on="days_in_program", 
    direction="nearest", 
    tolerance=30.0)

abx_diet.to_csv("../data/abx_covar_diet_merged.csv", index=False)

# Drop sparse col
threshold = 100
nan_counts = abx_diet.isna().sum()
columns_to_drop = nan_counts[nan_counts > threshold].index.tolist()
# Drop the columns
abx_diet = abx_diet.drop(columns=columns_to_drop)
abx_diet = abx_diet.drop(columns='response_percentage')

col_to_rename = abx_diet.iloc[:,29:].columns.tolist()
new_col_names = {col: col.split(':')[2] for col in col_to_rename}
abx_diet.rename(columns=new_col_names, inplace=True)
abx_diet.rename(columns={'achohol_type': 'alcohol_type', 
                         'cruciferous_vegetables':'cruciferous_vegetables_frequency',
                         'fruits':'fruits_per_day',
                         'vegetables':'vegetables_per_day',
                         'sugary_drinks':'sugary_drinks_per_day',
                         'water':'cups_water_per_day', 
                         'grains':'grains_per_day'}, inplace=True)
abx_diet.columns

# Change some longer text for plot below
shortened = {
    'cruciferous_vegetables_frequency': {'(1) 3-4 times per week': '(1) 3-4x/week', 
                            '(4) Less than once per month': '(4) > 1/month',
                            '(2) Once per week':'(2) 1/week', 
                            '(3) Once per month':'(3) 1/month'},
    'fruits_per_day': {'(0) Zero/less than 1 per day': '(0) 0'},
    'vegetables_per_day': {'(0) Zero/less than 1 per day': '(0) 0'}, 
    'sugary_drinks_per_day': {'(0) Zero/less than 1 per day': '(0) 0'}, 
    'cups_water_per_day': {'(0) I don’t usually drink water': '(0) None', 
                           '(1) Less than 3 cups per day':'(1) > 3 cups', 
                           '(4) More than 8 cups per day':'(4) 8+ cups',
                           '(2) 4-5 cups per day':'(2) 4-5 cups',
                           '(3) 6-8 cups per day':'(2) 6-8 cups'},
    'grains_per_day': {'(0) Zero/less than 1 per day': '(0) 0', '(4) 7 or more':'(4) 7+'}                     
  }
    
abx_diet.replace(shortened, inplace=True)

def map_survey_responses(value, mapping_dict):
    """Map a survey response to a category based on a dictionary, keeping NaNs intact."""
    if pd.isna(value):  # Ensure NaNs remain as NaNs
        return np.nan  
    
    value = str(value)  # Convert to string for dictionary lookup
    
    for key, category in mapping_dict.items():
        if value in key:
            return category
    return value  # Return original value if not in the dictionary

column_mappings_binary = {
    'fruits_per_day': {('(0) 0'): 0,
                       ('(1) 1', '(2) 2-3', '(3) 4-5', '(4) 6 or more'): 1},
    'vegetables_per_day': {('(0) 0'): 0,
                           ('(1) 1', '(2) 2-3', '(3) 4-5', '(4) 6 or more'): 1},
    'sugary_drinks_per_day': {('(0) 0'): 0,
                              ('(1) 1', '(2) 2-3', '(3) 4-5', '(4) 6 or more'): 1},
    'alcohol_drinks_a_day': {('(0) None'): 0, 
                             ('(1) 1-2 drinks', '(2) 3-4 drinks','(3) 5-6 drinks','(4) 6+ drinks'): 1},
    'grains_per_day': {('(0) 0'): 0,
                       ('(1) 1-2', '(2) 3-4','(3) 5-6','(4) 7+'): 1}, 
    'cruciferous_vegetables_frequency': {('(4) > 1/month', '(3) 1/month', '(2) 1/week'): 0, 
                                ('(1) 3-4x/week','(0) Daily'): 1}, 
    'breakfast': {('(4) > 1/month', '(3) 1/month', '(2) 1/week'): 0, 
                  ('(1) 3-4x/week','(0) Daily'): 1},
    'snack': {('(4) > 1/month', '(3) 1/month', '(2) 1/week'): 0,
              ('(1) 3-4x/week','(0) Daily'): 1}
    }

# Apply the mapping to each column
for col, mapping_dict in column_mappings_binary.items():
    abx_diet[col + '_binary'] = abx_diet[col].apply(map_survey_responses, args=(mapping_dict,))
abx_diet.columns

diet_cols = ['fruits_per_day_binary','vegetables_per_day_binary', 
             'sugary_drinks_per_day_binary','alcohol_drinks_a_day_binary', 
             'grains_per_day_binary','cruciferous_vegetables_frequency_binary',
             'alcohol_type','snack_binary', 'breakfast_binary']

shannon_binary = run_diet_regressions(diet_cols, abx_diet, "shannon_index")
features_binary = run_diet_regressions(diet_cols, abx_diet, "observed_features")
bp_binary = run_diet_regressions(diet_cols, abx_diet, "berger_parker")

# For supplement:
shannon_binary.to_csv("../data/diet_binary_shannon.csv", index=False)
features_binary.to_csv("../data/diet_binary_observed_features.csv", index=False)
bp_binary.to_csv("../data/diet_binary_berger_parker.csv", index=False)
