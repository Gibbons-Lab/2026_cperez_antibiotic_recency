import pandas as pd
import numpy as np
from rich.progress import track
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from patsy import dmatrices
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.get_chrome()

def get_results(feature, df, formula_template, term_of_interest):
    """
    Get table of features, beta, t-stat, p, and n for a given covariate.
    Note that if the term of interest is categorical, you must provide the 
    level in order to extract it ("C(meds_cholesterol)[T.Yes]")
    """
    formula = formula_template.format(feature=feature)
    fitted = ols(formula, data=df).fit()
    
    return pd.DataFrame({
        "feature": feature,
        "beta": fitted.params[term_of_interest],
        "se": fitted.bse[term_of_interest],
        "t_statistic": fitted.tvalues[term_of_interest],
        "conf_low": fitted.conf_int()[0][term_of_interest],
        "conf_high": fitted.conf_int()[1][term_of_interest],
        "p": fitted.pvalues[term_of_interest],
        "n": fitted.nobs
    }, index=[feature])

def run_tests(features, df, formula_template, term_of_interest):
    """
    Puts everything together to run our linear regression. In particular, 
    allows us to use track to see progress and adds FDR correction.
    """
    results = map(lambda f: get_results(f, df, formula_template, term_of_interest), track(features))
    results = list(results)
    combined = pd.concat(results)
    combined["q"] = multipletests(combined.p, method="fdr_bh")[1]
    return combined

def get_results_two_sets(formula, term, df):
    """
    Extracts the data for use cases where we are iterating through
    two sets of features (i.e., both microbe and metabolites).
    """
    try:
        fitted = ols(formula, data=df).fit()
        return {
            "term": term,
            "beta": fitted.params.get(term, float("nan")),
            "t": fitted.tvalues.get(term, float("nan")),
            "p": fitted.pvalues.get(term, float("nan")),
            "n": fitted.nobs
        }
    except Exception as e:
        print(f"Error fitting formula: {formula}\n→ {e}")
        return {
            "term": term,
            "beta": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "n": float("nan")
        }

def run_tests_two_sets(features, feature_set2, df, formula_template):
    all_results = []

    for outcome in track(features, description="Running regressions"):
        for predictor in feature_set2:
            formula = formula_template.format(feature=outcome, feature_set2=predictor)
            result = get_results_two_sets(formula, predictor, df)
            result["outcome_feature"] = outcome
            result["predictor"] = predictor
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    if not results_df.empty and results_df["p"].notna().any():
        results_df["q"] = multipletests(results_df["p"].fillna(1), method="fdr_bh")[1]
    else:
        results_df["q"] = float("nan")
    return results_df

def run_tests_paired(change_scores, baselines, df, formula_template, term):
    all_results = []

    for outcome, baseline in zip(change_scores, baselines):
        formula = formula_template.format(
            change_score=outcome,
            baseline=baseline
        )

        result = get_results_two_sets(formula, term, df)
        result["outcome_feature"] = outcome
        result["baseline"] = baseline
        result["predictor"] = term

        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    if not results_df.empty and results_df["p"].notna().any():
        results_df["q"] = multipletests(
            results_df["p"].fillna(1),
            method="fdr_bh"
        )[1]
    else:
        results_df["q"] = float("nan")

    return results_df

def get_logit_results(feature, formula_template, df, term_of_interest):
    """Fit logistic regression and return coef, OR, CI, p, q, n."""
    # Build formula and matrices
    formula = formula_template.format(feature=feature)
    y, X = dmatrices(formula, data=df, return_type='dataframe')
    
    # Fit logistic regression
    mod = sm.Logit(y, X)
    res = mod.fit(disp=0) # disp=0 hides optimization output
    
    # Extract values for term of interest
    coef = res.params[term_of_interest]
    OR = np.exp(coef)
    
    conf = res.conf_int().loc[term_of_interest]
    OR_conf = np.exp(conf)
    
    pval = res.pvalues[term_of_interest]
    
    return pd.DataFrame({
        "feature": [feature],
        "coef": [coef],
        "OR": [OR],
        "conf_low": [conf[0]],
        "conf_high": [conf[1]],
        "OR_conf_low": [OR_conf[0]],
        "OR_conf_high": [OR_conf[1]],
        "p": [pval],
        "n": [int(res.nobs)]
    }).set_index("feature")


def run_logit_tests(features, df, formula_template, term_of_interest):
    """Run logistic regression for many features and return combined results using
    the get_logit_results function."""
    results = [get_logit_results(f, formula_template, df, term_of_interest) for f in features]
    combined = pd.concat(results)
    
    # Add BH-FDR correction
    combined["q"] = multipletests(combined.p, method="fdr_bh")[1]
    
    return combined

def run_diet_regressions(col_to_iterate, data, outcome_var):
    """
    This function is a version of the regression code where we can (1) change
    the outcome variable to be which ever diversity metric desired, and (2)  
    iterate through diet variables (or anything else we think interact with
    antibiotic use). The interaction terms specifically are pulled out and returned
    as a dataframe.
    """
    all_results = []
    
    for food_bin in col_to_iterate:
        formula = (f"{outcome_var} ~ C(sex) + age + BMI_CALC + C(abx_recency)*C({food_bin}) + C(diet_vendor) + " 
                   f"C(meds_cholesterol) + C(meds_blood_sugar) + PC1 + PC2 + PC3 + PC4 + PC5 + centered_eGFR + C(vendor_dashboard)")
        fitted = ols(formula, data=data).fit()

        interaction_terms = [term for term in fitted.pvalues.index if "C(abx_recency)" in term and f"{food_bin}" in term]

        for term in interaction_terms:
            all_results.append({
                "food_bin": food_bin,
                "term": term,
                "beta": fitted.params[term],
                "t_stat": fitted.tvalues[term],
                "p_value": fitted.pvalues[term],
                "n": fitted.nobs
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        # Apply FDR correction across all tests
        results_df["q_value"] = multipletests(results_df["p_value"], method='fdr_bh')[1]


    return results_df

def run_year_interaction_regressions(col_to_iterate, data, outcome_var, categorical_pred = True):
    """
    This function is a verison of the regression code where we can (1) change
    the outcome variable to be which ever diversity metric desired, and (2)  
    iterate through diet variables (or anything else we think interact with
    antibiotic use). The interaction terms specifically are pulled out and returned
    as a dataframe. Similar to diet, only this is adapted to combine the antibiotic 
    groups for larger "N" when looking at sparse bugs and metabolites. For ONE outcome variable
    so that FDR correction is applied correctly.
    """
    all_results = []

    if categorical_pred == False:
        for cont_variable in col_to_iterate:
            formula = (f"{outcome_var} ~ C(sex) + age + BMI_CALC + C(abx_year)*{cont_variable} +" 
                       f"C(meds_cholesterol) + C(meds_blood_sugar) + PC1 + PC2 + PC3 + PC4 + PC5 + centered_eGFR")
            fitted = ols(formula, data=data).fit()
    
            interaction_terms = [term for term in fitted.pvalues.index if "C(abx_year)" in term and f"{cont_variable}" in term]
    
            for term in interaction_terms:
                all_results.append({
                    "continuous variable": cont_variable,
                    "term": term,
                    "beta": fitted.params[term],
                    "t_stat": fitted.tvalues[term],
                    "p_value": fitted.pvalues[term],
                    "n": fitted.nobs
                })
    else:   
        for microbe in col_to_iterate:
            formula = (f"{outcome_var} ~ C(sex) + age + BMI_CALC + C(abx_year)*C({microbe}) +" 
                       f"C(meds_cholesterol) + C(meds_blood_sugar) + PC1 + PC2 + PC3 + PC4 + PC5 + centered_eGFR + C(vendor_dashboard)")
            fitted = ols(formula, data=data).fit()
    
            interaction_terms = [term for term in fitted.pvalues.index if "C(abx_year)" in term and f"{microbe}" in term]
    
            for term in interaction_terms:
                all_results.append({
                    "microbe": microbe,
                    "term": term,
                    "beta": fitted.params[term],
                    "t_stat": fitted.tvalues[term],
                    "p_value": fitted.pvalues[term],
                    "n": fitted.nobs
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty: 
        # Apply FDR correction across all tests
        results_df["q_value"] = multipletests(results_df["p_value"], method='fdr_bh')[1]
    return results_df

def make_barbell_plot(metabolite, data_raw, data_tests):
    delta_none_abx_year = (data_raw[data_raw['abx_year'] == 0][metabolite].mean() - 
                           data_raw[data_raw['abx_year'] == 1][metabolite].mean())
    sig_metab = data_tests[data_tests.metabolite == metabolite].sort_values(by='beta')
    metab_name = sig_metab['CHEMICAL_NAME'].tolist()[0]
    barbell_data_absent = sig_metab[['plot_name','microbe']].copy()
    barbell_data_present = sig_metab[['plot_name','microbe']].copy()
    barbell_data_present['in_gut'] = 'present'
    barbell_data_absent['in_gut'] = 'absent'
    
    # Calculate differences for when bugs are present
    abx_deltas_list = []
    for bug in barbell_data_present.microbe.tolist():
        bug_present = data_raw[data_raw[bug] == 1]
        delta_calc = (bug_present[bug_present['abx_year'] == 1][metabolite].mean() - 
                      bug_present[bug_present['abx_year'] == 0][metabolite].mean())
        abx_deltas_list.append(delta_calc)    
    barbell_data_present['mean_delta_abx'] = abx_deltas_list
    
    # Calculate differences for when bugs are absent
    abx_deltas_list = []
    for bug in barbell_data_absent.microbe.tolist():
        bug_absent = data_raw[data_raw[bug] == 0]
        delta_calc = (bug_absent[bug_absent['abx_year'] == 1][metabolite].mean() - 
                  bug_absent[bug_absent['abx_year'] == 0][metabolite].mean())
        abx_deltas_list.append(delta_calc)
    barbell_data_absent['mean_delta_abx'] = abx_deltas_list

    barbell_data = pd.concat([barbell_data_present, barbell_data_absent], axis=0)

        # Split by presence_absence
    df_black = barbell_data[barbell_data["in_gut"] == "present"]   # adjust label if different
    df_white = barbell_data[barbell_data["in_gut"] == "absent"]
    
    fig = go.Figure()
    
    # Add black markers
    fig.add_trace(go.Scatter(
        x=df_black["mean_delta_abx"],
        y=df_black["plot_name"],
        mode="markers",
        marker=dict(color="black"),
        name="present"
    ))
    
    # Add white markers with black outline
    fig.add_trace(go.Scatter(
        x=df_white["mean_delta_abx"],
        y=df_white["plot_name"],
        mode="markers",
        marker=dict(color="white", line=dict(width=2, color="black")),
        name="absent"
    ))

    order = df_black.sort_values("mean_delta_abx")["plot_name"].tolist()

    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=order   # enforce this custom order
        )
    )
    
    # Add barbell lines
    for i in barbell_data["plot_name"].unique():
        df_sub = barbell_data[barbell_data["plot_name"] == i]
        fig.add_shape(
            type="line",
            layer="below",   # so the dots cap the line
            y0=df_sub.plot_name.values[0], x0=df_sub.mean_delta_abx.values[0],
            y1=df_sub.plot_name.values[1], x1=df_sub.mean_delta_abx.values[1],
            line=dict(color="black", width=2)
        )
    
    fig.update_layout(
        height= 110 + (20 * len(barbell_data_absent)),
        width= 700,
        title=f"{metab_name}",
        xaxis_title=f"Change in mean Z-score of log({metab_name}): users − non-users",
        yaxis_title="Taxon",
        margin=dict(l=80, r=40, t=80, b=40))
    
    fig.update_traces(
        marker=dict(size=9),          # adjust this number
        selector=dict(mode="markers")  # apply only to the markers, not the lines
    )
    
    fig.update_layout(
        plot_bgcolor="white",   # background color
        xaxis=dict(
            showgrid=True, 
            gridcolor="lightgrey",  # softer gridlines
            gridwidth=0.2
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="lightgrey",
            gridwidth=0.5
        )
    )
    
    fig.update_xaxes(
        zeroline=True, 
        zerolinecolor="darkgrey", # color it consistently with gridlines
        zerolinewidth=2     
    )
    
    fig.write_image(f"figures/microbe_metab_{metabolite}_interactions_mean_difference.pdf", width=700, height=110 + (20 * len(barbell_data_absent))) 
    fig.show() 