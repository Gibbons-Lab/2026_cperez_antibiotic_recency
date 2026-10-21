import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
sys.path.append(os.path.abspath(os.path.join('..')))
from regression_functions import fast_interaction_screen

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

# ---- One work item per chemistry; include_vendor decided by group (your #1) ----
# Single-vendor labs -> vendor is constant -> DROP C(vendor).
# Shared labs        -> both vendors measured it -> KEEP C(vendor).
jobs = ([(c, False) for c in not_offered_quest] +
        [(c, False) for c in not_offered_labcorp] +
        [(c, True) for c in shared_labs])

def _run_one(chemistry, include_vendor):
    # Set robust=True below to use HC3 heteroskedasticity-robust SEs.
    temp = fast_interaction_screen(metabolite_features, labs_mets, chemistry,
                                   include_vendor=include_vendor, robust=False)
    if not temp.empty:
        temp['chemistry'] = chemistry
    return temp

results = Parallel(n_jobs=4, backend="loky", verbose=5)(
    delayed(_run_one)(chem, inc) for chem, inc in jobs
)

all_metab_abx_interactions = pd.concat([r for r in results if not r.empty], ignore_index=True)

# q_value        = within-chemistry BH  (each lab is its own hypothesis family)
# q_value_global = BH across every interaction test in the entire run
all_metab_abx_interactions["q_value_global"] = multipletests(
    all_metab_abx_interactions["p_value"], method="fdr_bh"
)[1]

all_metab_abx_interactions.to_csv("../data/metabolite_abx_interactions_on_labs.csv", index=False)
print(f"Done. {len(all_metab_abx_interactions)} interaction tests across "
      f"{all_metab_abx_interactions['chemistry'].nunique()} chemistries.")
