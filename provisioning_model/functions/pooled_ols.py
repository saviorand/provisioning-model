import numpy as np
import statsmodels.api as sm
from scipy.stats import shapiro, spearmanr
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox

def analyze_regression_results(model_results, assumptions):
    """
    Analyze the regression results and print the outcomes.

    Parameters:
    model_results: The results of a fitted OLS model.
    assumptions: A dictionary containing analysis assumptions such as p-value thresholds.
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
        print(f"The F-test indicates that the parameter coefficients are jointly significant at p < {assumptions['p_value_threshold']}.")

    # Normality of Residuals
    w, p_value_normality = shapiro(model_results.resid)
    if p_value_normality < assumptions['p_value_threshold']:
        print("The residual errors of the model are not normally distributed, implying that the standard errors and confidence intervals associated with the model’s predictions may not be entirely reliable.")

    # Heteroskedasticity Test
    bp_test_stat, p_value_het, _, _ = het_breuschpagan(model_results.resid, model_results.model.exog)
    if p_value_het < assumptions['p_value_threshold']:
        print("The residual errors are heteroskedastic, implying that results of t-test for parameter significance, the corresponding confidence intervals for the parameter estimates, and the results of the F-test are not entirely reliable.")

    # Correlation of Residuals with Response Variable
    _, p_value_corr = spearmanr(model_results.resid, model_results.model.endog)
    if p_value_corr < assumptions['p_value_threshold']:
        print("The residual errors are correlated with the response variable y, which means the model may have left out important regression variables.")

    # Autocorrelation in Residuals
    autocorr_results = acorr_ljungbox(model_results.resid, lags=[1, 2, 3], return_df=True)
    significant_lags = autocorr_results.index[autocorr_results['lb_pvalue'] < assumptions['p_value_threshold']].tolist()

    if significant_lags:
        significant_lags_str = ', '.join(map(str, significant_lags))
        print(f"The residual errors are auto-correlated at lags {significant_lags_str}, implying a general miss-specification of the regression model.")
    else:
        print("No significant autocorrelation detected in the residual errors.")



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
