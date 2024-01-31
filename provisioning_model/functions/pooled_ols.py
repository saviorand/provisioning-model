import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
# from linearmodels.panel import PanelOLS, RandomEffects
from scipy.stats import shapiro, spearmanr
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from termcolor import colored


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
    df_filtered_years = df[df['TIME_PERIOD'].isin(years)]

    # Group by country and check if each has the same number of unique years as the length of the years list
    valid_countries = df_filtered_years.groupby('geo').filter(lambda x: x['TIME_PERIOD'].nunique() == len(years))['geo'].unique()

    balanced_panel_df = df_filtered_years[df_filtered_years['geo'].isin(valid_countries)]

    return balanced_panel_df


def analyze_regression_results(model_results, assumptions):
    """
    Analyze the regression results and print the outcomes.

    Parameters:
    model_results: The results of a fitted OLS model.
    assumptions: A dictionary containing analysis assumptions such as p-value thresholds.

    Returns:
    dict: A dictionary containing goodness-of-fit measures.
    """

    print("Analysis of Regression Results:")

    # Adjusted R-squared
    adj_r_squared = model_results.rsquared_adj
    print(f"Adjusted R-squared is around {adj_r_squared:.2%}.")

    # Significance of Model Coefficients
    p_values = model_results.pvalues
    significant_pvalues = p_values[p_values < assumptions['p_value_threshold']]
    if not significant_pvalues.empty:
        print(f"The following coefficients are significant at p < {assumptions['p_value_threshold']}:")
        print(significant_pvalues)

    # F-test for Joint Significance
    f_pvalue = model_results.f_pvalue
    if f_pvalue < assumptions['p_value_threshold']:
        print(
            f"The F-test indicates that the parameter coefficients are jointly significant at p < {assumptions['p_value_threshold']}.")

    # Normality of Residuals
    w, p_value_normality = shapiro(model_results.resid)
    if p_value_normality < assumptions['p_value_threshold']:
        print(colored('The residual errors of the model are not normally distributed', 'red'))
    else:
        print(colored('The residual errors are normally distributed', 'green'))

    # Heteroskedasticity Test
    bp_test_stat, p_value_het, _, _ = het_breuschpagan(model_results.resid, model_results.model.exog)
    if p_value_het < assumptions['p_value_threshold']:
        print(colored('The residual errors are heteroskedastic', 'red'))
    else:
        print(colored('The residual errors are homoskedastic', 'green'))

    # Correlation of Residuals with Response Variable
    _, p_value_corr = spearmanr(model_results.resid, model_results.model.endog)
    if p_value_corr < assumptions['p_value_threshold']:
        print(colored('The residual errors are correlated with the response variable y', 'red'))
    else:
        print(colored('The residual errors are not correlated with the response variable y', 'green'))

    # Autocorrelation in Residuals
    autocorr_results = acorr_ljungbox(model_results.resid, lags=[1, 2, 3], return_df=True)
    significant_lags = autocorr_results.index[autocorr_results['lb_pvalue'] < assumptions['p_value_threshold']].tolist()

    if significant_lags:
        significant_lags_str = ', '.join(map(str, significant_lags))
        print(colored(f"The residual errors are auto-correlated at lags {significant_lags_str}.", 'red'))
    else:
        print(colored('No significant autocorrelation detected in the residual errors.', 'green'))

    goodness_of_fit_measures = {
        'adj_r_squared': adj_r_squared,
        'log_likelihood': model_results.llf,
        'AIC': model_results.aic,
        # 'p_value_normality': p_value_normality,
        # 'p_value_het': p_value_het,
        # 'p_value_corr': p_value_corr,
        # 'significant_lags': significant_lags
    }
    return goodness_of_fit_measures


def pooled_ols_reg(regression_df, y_variable, x_variables, interaction_terms=[]):
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

    # Error handling for input variables
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
                pooled_x[':'.join(term)] = np.prod([pooled_x[item] for item in term], axis=0)
            else:
                raise ValueError(f"Interaction term {term} is not valid")

    pooled_olsr_model = sm.OLS(endog=pooled_y, exog=pooled_x)
    return pooled_olsr_model.fit()


def regression_model(regression_df, y_variable, x_variables, model_type='pooled_ols', interaction_terms=[], time_effects=False, unit_col_name='geo',  time_col_name='TIME_PERIOD'):
    """
    Perform a regression analysis.

    Parameters:
    regression_df (DataFrame): The dataframe containing the data.
    y_variable (str): The dependent variable.
    x_variables (list): The independent variables.
    model_type (str): Type of model ('pooled_ols' or 'fixed_effects').
    interaction_terms (list of tuples): Optional interaction terms.
    unit_col_name (str): The column name for unit (entity) in fixed effects model.

    Returns:
    RegressionResults: The fitted model's results.
    """
     # Pooled OLS Regression
    if model_type == 'pooled_ols':
        pooled_x = regression_df[x_variables].replace([np.inf, -np.inf], np.nan).dropna()
        pooled_y = regression_df[y_variable].replace([np.inf, -np.inf], np.nan).dropna()
        pooled_y = pooled_y[pooled_y.index.isin(pooled_x.index)]
        pooled_x = pooled_x[pooled_x.index.isin(pooled_y.index)]
        pooled_x = sm.add_constant(pooled_x)

        if interaction_terms:
            for term in interaction_terms:
                if all(item in pooled_x.columns for item in term):
                    pooled_x[':'.join(term)] = np.prod([pooled_x[item] for item in term], axis=0)
                else:
                    raise ValueError(f"Interaction term {term} is not valid")

        model = sm.OLS(endog=pooled_y, exog=pooled_x)
        return model.fit()

    # Fixed Effects Regression
    elif model_type == 'fixed_effects':
        if unit_col_name is None:
            raise ValueError("unit_col_name is required for fixed effects model")

        # Create dummies for fixed effects
        dummies = pd.get_dummies(regression_df[unit_col_name], drop_first=True)
        regression_df_with_dummies = regression_df.join(dummies)

        # Ensure the independent variables list does not include the unit_col_name
        x_variables = [var for var in x_variables if var != unit_col_name]

        # Build the regression formula
        fe_expr = f'{y_variable} ~ {" + ".join(x_variables)}'
        for dummy in dummies.columns[:-1]:  # Exclude the last dummy
            fe_expr += f' + {dummy}'
        
        # Fit the fixed effects model
        fe_model = smf.ols(formula=fe_expr, data=regression_df_with_dummies)
        return fe_model.fit()
