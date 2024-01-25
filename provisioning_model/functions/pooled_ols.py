import numpy as np
import statsmodels.api as sm


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
