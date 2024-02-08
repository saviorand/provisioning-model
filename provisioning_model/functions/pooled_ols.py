import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import PooledOLS
from linearmodels.panel import PanelOLS, RandomEffects
from scipy.stats import shapiro, spearmanr
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    acorr_ljungbox,
    het_white,
    het_breuschpagan,
)
from statsmodels.stats.stattools import durbin_watson
from termcolor import colored
from IPython.display import display, Markdown


def create_balanced_panel(df, years):
    """
    Creates a balanced panel DataFrame including only the countries
    that have observations for all the specified years and only for those years.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    years (list): A list of years.

    Returns:
    pandas.DataFrame: A balanced panel DataFrame.
    """
    df_filtered_years = df[df["TIME_PERIOD"].isin(years)]

    # Group by country and check if each has the same number of unique years as the length of the years list
    valid_countries = (
        df_filtered_years.groupby("geo")
        .filter(lambda x: x["TIME_PERIOD"].nunique() == len(years))["geo"]
        .unique()
    )

    balanced_panel_df = df_filtered_years[
        df_filtered_years["geo"].isin(valid_countries)
    ]

    return balanced_panel_df


def analyze_regression_results(model_results, assumptions, model_name="OLS"):
    """
    Analyze the regression results and print the outcomes.

    Parameters:
    model_results: The results of a fitted OLS model.
    assumptions: A dictionary containing analysis assumptions such as p-value thresholds.

    Returns:
    dict: A dictionary containing goodness-of-fit measures.
    """

    # print("Analysis of Regression Results:")

    # Adjusted R-squared
    adj_r_squared = model_results.rsquared_adj
    # print(f"Adjusted R-squared is around {adj_r_squared:.2%}.")

    # Significance of Model Coefficients
    p_values = model_results.pvalues
    significant_pvalues = p_values[p_values < assumptions["p_value_threshold"]]
    # if not significant_pvalues.empty:
    #     print(
    #         f"The following coefficients are significant at p < {assumptions['p_value_threshold']}:"
    #     )
    #     print(significant_pvalues)

    # F-test for Joint Significance
    f_pvalue = model_results.f_pvalue
    # if f_pvalue < assumptions["p_value_threshold"]:
    #     print(
    #         f"The F-test indicates that the parameter coefficients are jointly significant at p < {assumptions['p_value_threshold']}."
    #     )

    # Normality of Residuals
    w, p_value_normality = shapiro(model_results.resid)
    # if p_value_normality < assumptions["p_value_threshold"]:
    #     print(
    #         colored(
    #             "The residual errors of the model are not normally distributed", "red"
    #         )
    #     )
    # else:
    #     print(colored("The residual errors are normally distributed", "green"))

    # Heteroskedasticity Test
    bp_test_stat, p_value_het, _, _ = het_breuschpagan(
        model_results.resid, model_results.model.exog
    )
    # if p_value_het < assumptions["p_value_threshold"]:
    #     print(colored("The residual errors are heteroskedastic", "red"))
    # else:
    #     print(colored("The residual errors are homoskedastic", "green"))

    # Correlation of Residuals with Response Variable
    _, p_value_corr = spearmanr(model_results.resid, model_results.model.endog)
    # if p_value_corr < assumptions["p_value_threshold"]:
    #     print(
    #         colored(
    #             "The residual errors are correlated with the response variable y", "red"
    #         )
    #     )
    # else:
    #     print(
    #         colored(
    #             "The residual errors are not correlated with the response variable y",
    #             "green",
    #         )
    #     )

    # Autocorrelation in Residuals
    autocorr_results = acorr_ljungbox(
        model_results.resid, lags=[1, 2, 3], return_df=True
    )
    significant_lags = autocorr_results.index[
        autocorr_results["lb_pvalue"] < assumptions["p_value_threshold"]
    ].tolist()

    # if significant_lags:
    #     significant_lags_str = ", ".join(map(str, significant_lags))
    #     print(
    #         colored(
    #             f"The residual errors are auto-correlated at lags {significant_lags_str}.",
    #             "red",
    #         )
    #     )
    # else:
    #     print(
    #         colored(
    #             "No significant autocorrelation detected in the residual errors.",
    #             "green",
    #         )
    #     )

    goodness_of_fit_measures = {
        "model_name": model_name,
        "adj_r_squared": adj_r_squared,
        "log_likelihood": model_results.llf,
        "AIC": model_results.aic,
        # 'p_value_normality': p_value_normality,
        # 'p_value_het': p_value_het,
        # 'p_value_corr': p_value_corr,
        # 'significant_lags': significant_lags
    }
    return goodness_of_fit_measures


def regression_model_statsmodels(
    regression_df,
    y_variable,
    x_variables,
    model_type="pooled_ols",
    interaction_terms=[],
):
    """
    Perform a pooled OLS regression.

    Parameters:
    regression_df (DataFrame): The dataframe containing the data.
    y_variable (str): The dependent variable.
    x_variables (list): The independent variables.
    interaction_terms (list of tuples): Optional interaction terms (each term as a tuple of variables).

    Returns:
    RegressionResults: The fitted model's results.
    """

    # print(
    #     colored(
    #         f"### {model_type} with {y_variable} as the dependent variable and {x_variables} as independent variables. statsmodels method ###",
    #         "green",
    #     )
    # )
    # heading = Markdown(
    #     f"### {model_type} with {y_variable} as the dependent variable and {x_variables} as independent variables. statsmodels method"
    # )
    # display(heading)

    for var in [y_variable] + x_variables:
        if var not in regression_df.columns:
            raise ValueError(f"{var} is not in the DataFrame")

    pooled_x = regression_df[x_variables].replace([np.inf, -np.inf], np.nan).dropna()
    pooled_y = regression_df[y_variable].replace([np.inf, -np.inf], np.nan).dropna()
    pooled_y = pooled_y[pooled_y.index.isin(pooled_x.index)]
    pooled_x = pooled_x[pooled_x.index.isin(pooled_y.index)]

    pooled_x = sm.add_constant(pooled_x)

    if interaction_terms:
        for term in interaction_terms:
            if all(item in pooled_x.columns for item in term):
                pooled_x[":".join(term)] = np.prod(
                    [pooled_x[item] for item in term], axis=0
                )
            else:
                raise ValueError(f"Interaction term {term} is not valid")

    pooled_olsr_model = sm.OLS(endog=pooled_y, exog=pooled_x)
    pooled_olsr_model_results = pooled_olsr_model.fit()

    if model_type == "pooled_ols":
        return pooled_olsr_model_results

    elif model_type == "fixed_effects" or model_type == "random_effects":
        unit_names = regression_df["geo"].unique()
        n = len(unit_names)
        # print("Number of groups=" + str(n))
        T = regression_df.shape[0] / n
        # print("Number of time periods per group=" + str(T))
        N = n * T
        # print("Total number of observations=" + str(N))
        k = len(x_variables) + 1
        # print("Number of regression variables=" + str(k))

        dummies = pd.get_dummies(regression_df["geo"], drop_first=True)
        regression_df_with_dummies = regression_df.join(dummies)

        x_variables = [var for var in x_variables if var != "geo"]

        fe_expr = f'{y_variable} ~ {" + ".join(x_variables)}'
        for dummy in dummies.columns[:-1]:  # Exclude the last dummy
            fe_expr += f" + {dummy}"

        fe_model = smf.ols(formula=fe_expr, data=regression_df_with_dummies)
        fe_model_results = fe_model.fit()

        if model_type == "random_effects":
            # Get $\sigma^2_\epsilon$ and $\sigma^2_{pooled}$ from the fixed effects model.
            # Calculate $\sigma^2_u$ as $\sigma^2_{pooled} - \sigma^2_\epsilon$
            fe_ssr = fe_model_results.ssr
            pooled_ols_ssr = pooled_olsr_model_results.ssr

            sigma2_epsilon = fe_ssr / (n * T - (n + k + 1))
            sigma2_pooled = pooled_ols_ssr / (n * T - (k + 1))
            sigma2_u = sigma2_pooled - sigma2_epsilon

            # Calculate group-specific means
            regression_df_group_means = regression_df.groupby("geo").mean()
            regression_df_group_means["const"] = 1.0

            theta = 1 - math.sqrt(sigma2_epsilon / (sigma2_epsilon + T * sigma2_u))

            # Subtract the group means multiplied by theta from the original data
            # For independent variables
            for column in x_variables:
                regression_df[column] -= (
                    (regression_df_group_means[column] * theta)
                    .reindex(regression_df["geo"])
                    .values
                )
            # For dependent variable
            regression_df[y_variable] -= (
                (regression_df_group_means[y_variable] * theta)
                .reindex(regression_df["geo"])
                .values
            )
            re_X = regression_df[x_variables]
            re_y = regression_df[y_variable]

            # Add a constant to the independent variables, if not already included
            re_X = sm.add_constant(re_X, prepend=False)

            # Ensure no NaN values in re_X and re_y before alignment
            re_X.dropna(inplace=True)
            re_y.dropna(inplace=True)

            # Aligning X and y based on non-missing indices - might not be necessary if both are already cleaned
            non_missing_indices = re_y.index.intersection(re_X.index)
            re_X_aligned = re_X.loc[non_missing_indices]
            re_y_aligned = re_y.loc[non_missing_indices]

            # Fit the model
            re_model = sm.OLS(endog=re_y_aligned, exog=re_X_aligned)
            re_model_results = re_model.fit()
            return re_model_results

        else:
            return fe_model_results


def regression_model_linearmodels(
    regression_df,
    y_variable,
    x_variables,
    model_type="pooled_ols",
    interaction_terms=[],
    time_effects=False,
):
    """
    Perform a regression analysis.

    Parameters:
    regression_df (DataFrame): The dataframe containing the data.
    y_variable (str): The dependent variable.
    x_variables (list): The independent variables.
    model_type (str): Type of model ('pooled_ols' or 'fixed_effects').
    interaction_terms (list of tuples): Optional interaction terms.
    time_effects (bool): Whether to include time effects.

    Returns:
    RegressionResults: The fitted model's results.
    """

    # heading = Markdown(
    #     f"### {model_type} with {y_variable} as the dependent variable and {x_variables} as independent variables. linearmodels method ###"
    # )
    # display(heading)

    dataset = regression_df.copy()
    dataset = dataset.set_index(["geo", "TIME_PERIOD"])
    # years = dataset.index.get_level_values("TIME_PERIOD").to_list()
    # dataset["year"] = pd.Categorical(years)
    exog = sm.add_constant(dataset[x_variables])
    endog = dataset[y_variable]

    if interaction_terms:
        for term in interaction_terms:
            if all(item in exog.columns for item in term):
                exog[":".join(term)] = np.prod([exog[item] for item in term], axis=0)
            else:
                raise ValueError(f"Interaction term {term} is not valid")

    # Pooled OLS Regression
    if model_type == "pooled_ols":
        pooled_model = PooledOLS(endog, exog)
        return pooled_model.fit(cov_type="clustered", cluster_entity=True)

    elif model_type == "fixed_effects":
        fe_model = PanelOLS(endog, exog, entity_effects=True)
        return fe_model.fit()

    elif model_type == "random_effects":
        re_model = RandomEffects(endog, exog)
        return re_model.fit()


def analyze_linearmodels_regression_results(
    model_results,
    assumptions,
    model_name="PanelOLS",
    x_variables=None,
    model_type="pooled_ols",
):
    """
    Analyze the regression results from linearmodels and print the outcomes.

    Parameters:
    model_results: The results of a fitted regression model from linearmodels.
    assumptions: A dictionary containing analysis assumptions such as p-value thresholds.

    Returns:
    dict: A dictionary containing goodness-of-fit measures.
    """

    # print("Analysis of Regression Results:")

    fittedvals = model_results.predict().fitted_values
    residuals = model_results.resids
    n = model_results.nobs
    k = model_results.params.shape[0]  # This includes the intercept, if present

    # Adjusted R squared
    if model_type == "pooled_ols" or model_type == "random_effects":
        r_squared = model_results.rsquared_overall
    elif model_type == "fixed_effects":
        r_squared = model_results.rsquared_within

    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))

    # 3A. Homoskedasticity
    # 3A.1 Residuals-Plot for growing Variance Detection
    # fig, ax = plt.subplots()
    # ax.scatter(fittedvals, residuals, color="blue")
    # ax.axhline(0, color="r", ls="--")
    # ax.set_xlabel("Predicted Values", fontsize=15)
    # ax.set_ylabel("Residuals", fontsize=15)
    # ax.set_title("Homoskedasticity Test", fontsize=30)
    # plt.show()

    # 3A.2 White-Test
    # pooled_OLS_dataset = pd.concat([dataset, residuals_pooled_OLS], axis=1)
    exog = sm.tools.tools.add_constant(x_variables).fillna(0)
    # white_test_results = het_white(residuals, exog)

    labels = ["LM-Stat", "LM p-val", "F-Stat", "F p-val"]
    # print("White-Test:", dict(zip(labels, white_test_results)))

    # 3A.3 Breusch-Pagan-Test
    # breusch_pagan_test_results = het_breuschpagan(residuals, exog)
    labels = ["LM-Stat", "LM p-val", "F-Stat", "F p-val"]
    # print("Breusch-Pagan-Test:", dict(zip(labels, breusch_pagan_test_results)))

    # 3.B Non-Autocorrelation
    # Durbin-Watson-Test
    durbin_watson_test_results = durbin_watson(residuals)
    # print("Durbin-Watson-Test:", durbin_watson_test_results)

    goodness_of_fit_measures = {
        "model_name": model_name,
        "adj_r_squared": adjusted_r_squared,
        "log_likelihood": model_results.loglik,
        "AIC": None,  # TODO
    }
    return goodness_of_fit_measures
