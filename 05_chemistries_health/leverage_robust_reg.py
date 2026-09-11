import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

import arivale_data_interface as adi
# apply settings so the lib knows where the data is
adi.apply_isb_config()

import re
import ast
from rich.progress import track
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.robust.norms as norms

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
# Now you can import your file
from regression_functions import *

# ---- Read + merge ------------
labs_clean = pd.read_csv("../data/clincal_chems_abx_users_clean.csv", dtype={'public_client_id': str})
metabolites_merged = pd.read_csv("../data/metabolites_merged_used_for_regressions.csv", dtype={'public_client_id': str})

# Grab vendor
quest = labs_clean[labs_clean['vendor'] == 'Quest']
labcorp = labs_clean[labs_clean['vendor'] == 'LCA']

# Prepare labs for merge
lab_features = labs_clean.columns[31:].tolist()
cols = lab_features + ['vendor']
labs_clean.set_index('public_client_id', inplace=True)
labs_for_merge = labs_clean[cols].reset_index()

# Merge and set year var
labs_mets = metabolites_merged.merge(labs_for_merge, on='public_client_id', how='left')
labs_mets['abx_year'] = labs_mets['abx_recency'].apply(lambda x: "none" if x == "none" else "past_year")

metabolite_features = metabolites_merged.columns[27:].tolist()
not_offered_quest = quest[lab_features].isna().all().index[quest[lab_features].isna().all()].tolist()
not_offered_labcorp = labcorp[lab_features].isna().all().index[labcorp[lab_features].isna().all()].tolist()
individual_vendors = not_offered_labcorp + not_offered_quest
shared_labs = list(set(lab_features) - set(individual_vendors))

interaction_results = pd.read_csv("../data/metabolite_abx_interactions_on_labs.csv")

sig_interactions = interaction_results[
(interaction_results['q_value'] < 0.1) & (interaction_results['q_value_global'] < 0.1)
].sort_values(by=['q_value_global','q_value'])

# Merge with annotations
anns_snapshot = pd.read_csv("../data/metabolites_snapshot_updated.csv")
anns_snapshot = anns_snapshot.rename(columns={'feature':'metabolite'})
sig_ints_anns = sig_interactions.merge(anns_snapshot, on='metabolite', how='left')

# Define covars
covariates = ["sex", "age", "BMI_CALC", "meds_cholesterol",
              "meds_blood_sugar", "PC1", "PC2", "PC3", "PC4", "PC5",
              "centered_eGFR", "vendor"]

def interaction_influence(df, lab, metabolite, covars,
                          abx="abx_year"):
    """Refit one lab~abx*metabolite+covars model, return influence on the
    interaction term + a leverage-dropped refit for comparison."""
    d = df[[lab, abx, metabolite] + covars].dropna().copy()
    n = len(d)

    rhs = f"C({abx}) * {metabolite}" + ("" if not covars else " + " + " + ".join(covars))
    fit = smf.ols(f"Q('{lab}') ~ {rhs}", data=d).fit()

    # find the interaction column in the design (patsy names it with ':')
    inter = [c for c in fit.params.index if ":" in c and metabolite in c]
    assert len(inter) == 1, f"interaction term not uniquely found: {inter}"
    inter = inter[0]

    infl = fit.get_influence()
    cooks = infl.cooks_distance[0]                       # (n,)
    dfb   = infl.dfbetas[:, list(fit.params.index).index(inter)]  # (n,) for the interaction

    flag = (np.abs(dfb) > 2/np.sqrt(n)) | (cooks > 4/n)

    # refit dropping the flagged high-influence points
    fit_drop = smf.ols(f"Q('{lab}') ~ {rhs}", data=d.loc[~flag]).fit()

    return {
        "lab": lab, "metabolite": metabolite, "n": n, "n_dropped": int(flag.sum()),
        "beta_full":  fit.params[inter],      "p_full":  fit.pvalues[inter],
        "beta_drop":  fit_drop.params[inter], "p_drop":  fit_drop.pvalues[inter],
        "max_abs_dfbetas": np.abs(dfb).max(),
        "max_cooks": cooks.max(),
        # the actual influential rows, for eyeballing
        "influential_idx": d.index[flag].tolist(),
    }

# run across the hits
lev_results = pd.DataFrame(
    interaction_influence(labs_mets, r.chemistry, r.metabolite, covariates)
    for r in sig_ints_anns.itertuples() 
)

lev_results.to_csv('../data/abx_metab_lab_interactions_leverage.csv', index=False)

def robust_regression(df, lab, metabolite, covars, abx="abx_year"):
    """Refit model using robust regression"""
    d = df[[lab, abx, metabolite] + covars].dropna().copy()
    n = len(d)

    rhs = f"C({abx}) * {metabolite}" + ("" if not covars else " + " + " + ".join(covars))
    fit_rlm = smf.rlm(f"Q('{lab}') ~ {rhs}", data=d, M=norms.HuberT()).fit()

    # find the interaction column in the design (patsy names it with ':')
    inter = [c for c in fit_rlm.params.index if ":" in c and metabolite in c]
    assert len(inter) == 1, f"interaction term not uniquely found: {inter}"
    inter = inter[0]
    
    return {
        "lab": lab,
        "metabolite": metabolite,
        "n": n,
        "beta_robust": fit_rlm.params[inter],
        "p_robust": fit_rlm.pvalues[inter]
    }

# run across the hits
results_robust = pd.DataFrame(
    robust_regression(labs_mets, r.chemistry, r.metabolite, covariates)
    for r in sig_ints_anns.itertuples() 
)

results_robust.to_csv('../data/abx_metab_lab_interactions_robust.csv', index=False)
