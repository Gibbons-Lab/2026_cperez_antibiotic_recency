import pandas as pd
import numpy as np

from rich.progress import track
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from scipy import stats
import os

from patsy import dmatrices
from patsy import dmatrix
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
#pio.get_chrome()


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
    iterate through diet variables (or anything else like matbs we think interact with
    antibiotic use). The interaction terms specifically are pulled out and returned
    as a dataframe. Similar to diet, only this is adapted to combine the antibiotic 
    groups for larger "N" when looking at sparse bugs and metabolites. For ONE outcome variable
    so that FDR correction is applied correctly.
    """
    all_results = []

    if categorical_pred == False:
        for cont_variable in col_to_iterate:
            formula = (f"{outcome_var} ~ C(sex) + age + BMI_CALC + C(abx_year)*{cont_variable} + "
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
            formula = (f"{outcome_var} ~ C(sex) + age + BMI_CALC + C(abx_year)*C({microbe}) + C(vendor_dashboard) +" 
                       f"C(meds_cholesterol) + C(meds_blood_sugar) + PC1 + PC2 + PC3 + PC4 + PC5 + centered_eGFR")
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
    
    fig.write_image(f"../figures/microbe_metab_{metabolite}_interactions_mean_difference.pdf", width=700, height=110 + (20 * len(barbell_data_absent)), engine=None) 
    fig.show() 


def fast_interaction_screen(metabolites, data, outcome_var,
                            include_vendor=True, min_n=30, robust=False):
    """
    Screen abx_year x metabolite interactions for ONE clinical-chemistry outcome.

    Numeric (lstsq) reimplementation of the OLS interaction screen. Reproduces
    statsmodels' coefficient / SE / t to numerical precision, but:
      - builds the fixed covariate design once, only swaps the 2 metabolite cols
      - identifies the interaction by POSITION (last column), so it is immune to
        the substring-name-collision bug (col_35 vs col_351)
      - drops single-vendor collinearity via include_vendor=False + a rank guard

    Complete-case per metabolite (matches ols default NaN handling); n travels
    with each row. Per-chemistry BH-FDR is applied here (q_value). The GLOBAL
    (cross-chemistry) FDR is applied by the caller after all chemistries finish.

    Parameters
    ----------
    metabolites : list[str]   metabolite column names to iterate (predictors)
    data : pd.DataFrame        merged labs+metabolites frame
    outcome_var : str          the clinical-chemistry column (outcome)
    include_vendor : bool      True for shared labs (both vendors measured it),
                               False for single-vendor labs (vendor is constant
                               -> C(vendor) would be perfectly collinear)
    min_n : int                skip a metabolite with fewer complete-case rows
    robust : bool              True -> HC3 heteroskedasticity-robust SEs

    Returns
    -------
    pd.DataFrame with columns: metabolite, beta, t_stat, p_value, n, q_value
    """
    vend = " + C(vendor)" if include_vendor else ""
    base_rhs = ("C(sex) + age + BMI_CALC + C(abx_year) + C(meds_cholesterol) + "
                "C(meds_blood_sugar) + PC1 + PC2 + PC3 + PC4 + PC5 + centered_eGFR" + vend)

    base_cols = ["abx_year", "sex", "BMI_CALC", "age", "meds_cholesterol",
                 "meds_blood_sugar", "PC1", "PC2", "PC3", "PC4", "PC5", "centered_eGFR"]
    if include_vendor:
        base_cols.append("vendor")

    rows = []
    for met in metabolites:
        sub = data[[outcome_var, met] + base_cols].dropna()
        if sub.shape[0] < min_n:
            continue

        Xb = np.asarray(dmatrix(base_rhs, sub, return_type="dataframe"), dtype=float)
        abx = (sub["abx_year"].values == "past_year").astype(float)
        m = sub[met].values.astype(float)
        X = np.column_stack([Xb, m, abx * m])      # ..., metabolite, INTERACTION (last)
        y = sub[outcome_var].values.astype(float)

        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        n, p = X.shape
        if rank < p:                                # collinear (e.g. constant vendor)
            continue
        resid = y - X @ beta
        dof = n - p
        XtX_inv = np.linalg.inv(X.T @ X)
        j = p - 1                                   # interaction column

        if robust:                                        # HC3 robust SE
            h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)   # leverages
            u = resid / (1.0 - h)
            meat = (X * (u ** 2)[:, None]).T @ X
            cov = XtX_inv @ meat @ XtX_inv
            se_j = np.sqrt(cov[j, j])
        else:
            sigma2 = resid @ resid / dof
            se_j = np.sqrt(sigma2 * XtX_inv[j, j])

        t = beta[j] / se_j
        rows.append({"metabolite": met, "beta": beta[j], "t_stat": t,
                     "p_value": 2 * stats.t.sf(abs(t), dof), "n": n})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = multipletests(out["p_value"], method="fdr_bh")[1]  # within-chemistry
    return out

# moderator palette: 0 = none (no recent abx), 1 = past_year
abx_year_pal = {0: "#007EA7", 1: "red"}

# black text + Helvetica everywhere (Liberation/Nimbus Sans substitute on Linux)
FONT = dict(family="Helvetica, Arial, sans-serif", color="black", size=13)

RECENCY_ORDER = ["none", "twelve", "three"]
RECENCY_LABEL = {"none": "none", "twelve": "12mo", "three": "3mo"}
abx_recency_pal = {'none': "#007ea7", 'twelve': "#80ced7", 'three': '#ccdbdc'}

def _adjusted_slope(sub, chem, metab, covariates, standardize=True, min_n=20):
    """Covariate-adjusted (optionally standardized) slope of chem ~ metab on one subset."""
    d = sub[[chem, metab] + _cols(covariates)].dropna()
    if d.shape[0] < min_n or d[chem].std() == 0 or d[metab].std() == 0:
        return np.nan
    if standardize:
        y = ((d[chem] - d[chem].mean()) / d[chem].std()).values
        x = ((d[metab] - d[metab].mean()) / d[metab].std()).values
    else:
        y, x = d[chem].values, d[metab].values
    rhs = " + ".join(covariates) if covariates else "1"
    Xc = np.asarray(dmatrix(rhs, d, return_type="dataframe"), dtype=float)
    X = np.column_stack([Xc, x])            # metab is the LAST column
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < X.shape[1]:                   # collinear -> unreliable
        return np.nan
    return beta[-1]

def _cols(covariates):
    """Strip C(...) wrappers to raw column names for dropna."""
    import re
    out = []
    for term in covariates:
        m = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", term)
        out.append(m[0] if m else term)
    return list(dict.fromkeys(out))

def _pooled_slopes(sub, chem, metab, covariates, recency_col, levels,
                   standardize=True, min_n=20):
    """Per-level metabolite->chemistry slopes from ONE pooled categorical fit.

    Returns {level: slope} where slope[levels[0]] is the baseline metab coef and
    the others are baseline + that level's interaction offset. None if degenerate.
    These are the SAME numbers is_monotonic/clean_gradient were computed from.
    """
    cols = [chem, metab, recency_col] + _cols(covariates)
    d = sub[cols].dropna()
    d = d[d[recency_col].isin(levels)].copy()
    if d.shape[0] < min_n or d[chem].std() == 0 or d[metab].std() == 0:
        return None
    if standardize:                                   # z-score ONCE across pooled sample
        d["_y"] = (d[chem] - d[chem].mean()) / d[chem].std()
        d["_x"] = d[metab] #(d[metab] - d[metab].mean()) / d[metab].std()
    else:
        d["_y"], d["_x"] = d[chem], d[metab]

    covar_rhs = (" + " + " + ".join(covariates)) if covariates else ""
    ref = levels[0]
    fit = smf.ols(
        f"_y ~ C({recency_col}, Treatment(reference='{ref}')) * _x {covar_rhs}",
        data=d,
    ).fit()

    base = fit.params.get("_x", np.nan)               # none-group (reference) slope
    slopes = {ref: base}
    for lvl in levels[1:]:
        inter = [c for c in fit.params.index
                 if ":" in c and "_x" in c and f"[T.{lvl}]" in c]
        off = fit.params[inter[0]] if inter else np.nan
        slopes[lvl] = base + off                       # baseline + offset
    return slopes
    
def make_recency_barbell(data_raw, data_tests, covariates,
                         recency_col="abx_recency", levels=RECENCY_ORDER,
                         standardize=True, name_col="CHEMICAL_NAME",
                         row_px=26, header_px=30, gap_px=16,
                         out_pdf="../figures/recency_barbell_combined.pdf"):
    """Combined figure across ALL final hits; dots = pooled-model slopes.

    data_tests : final gated hits (cols: chemistry, metabolite; optional name_col)
    Returns (fig, bar_df).
    """
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    raw = data_raw[data_raw[recency_col].isin(levels)].copy()

    rows = []
    for _, hit in data_tests.iterrows():
        chem, metab = hit["chemistry"], hit["metabolite"]
        label = hit[name_col] if name_col in data_tests.columns else metab
        is_ctrl = str(metab).lower() == str(chem).lower()
        row_label = f"{chem} | {label}" + (" (+ctrl)" if is_ctrl else "")

        slopes = _pooled_slopes(raw, chem, metab, covariates,
                                recency_col, levels, standardize)
        if slopes is None or any(pd.isna(list(slopes.values()))):
            continue                                   # need a full track
        for lvl in levels:
            rows.append({
                "chemistry": chem,
                "metabolite": metab,
                "row_label": row_label,
                "is_ctrl": is_ctrl,
                "level": lvl,
                "slope": slopes[lvl],
            })
    bar = pd.DataFrame(rows)
    if bar.empty:
        print("no hits with a full pooled-model track; check level labels / group sizes")
        return None, bar

    # row order: GROUP BY chemistry, then DESCENDING by the none-group slope
    ref = levels[0]
    none_slope = (bar[bar["level"] == ref]
                  .set_index("row_label")["slope"])
    meta = (bar[["chemistry", "row_label"]].drop_duplicates()
            .assign(none_slope=lambda t: t["row_label"].map(none_slope))
            .sort_values(["chemistry", "none_slope"], ascending=[True, False]))
    y_order = meta["row_label"].tolist()               # visual top -> bottom

    fig = go.Figure()

    # connecting polyline through the recency points (drawn first, dots on top)
    for rl in bar["row_label"].unique():
        s = bar[bar["row_label"] == rl].set_index("level").reindex(levels)
        xs = s["slope"].values
        for a, b in zip(range(len(levels) - 1), range(1, len(levels))):
            fig.add_shape(type="line", layer="below",
                          y0=rl, x0=xs[a], y1=rl, x1=xs[b],
                          line=dict(color="lightgrey", width=2))

    # one dot-trace per recency level; diamonds mark positive controls
    for lvl in levels:
        d = bar[bar["level"] == lvl]
        fig.add_trace(go.Scatter(
            x=d["slope"], y=d["row_label"], mode="markers",
            name=RECENCY_LABEL.get(lvl, lvl),
            marker=dict(size=10, color=abx_recency_pal.get(lvl, "grey"),
                        symbol=np.where(d["is_ctrl"], "diamond", "circle"),
                        line=dict(width=1, color="black")),
        ))

    slope_lab = "Standardized slope (pooled model)" if standardize else "Slope (pooled model)"
    h = 160 + 24 * len(y_order)
    fig.update_layout(
        height=h, width=820,
        title="Antibiotic-recency gradient in metabolite -> chemistry slopes",
        xaxis_title=f"{slope_lab} of chemistry ~ metabolite",
        yaxis_title="Chemistry | metabolite",
        plot_bgcolor="white", margin=dict(l=260, r=40, t=80, b=50),
        legend_title="abx_recency",
        font=FONT, title_font=FONT, legend=dict(font=FONT),
        # categoryarray is bottom->top, so reverse the visual top->bottom order
        yaxis=dict(categoryorder="array", categoryarray=y_order[::-1],
                   showgrid=True, gridcolor="lightgrey", gridwidth=0.5,
                   title_font=FONT, tickfont=FONT),
        xaxis=dict(showgrid=True, gridcolor="lightgrey", gridwidth=0.2,
                   zeroline=True, zerolinecolor="darkgrey", zerolinewidth=2,
                   title_font=FONT, tickfont=FONT),
    )

    fig.write_image(out_pdf, width=820, height=h, engine=None)
    print(f"wrote {out_pdf}  ({len(y_order)} hits, "
          f"{bar['chemistry'].nunique()} chemistries)")
    return fig, bar