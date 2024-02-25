import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def inverse_scale_variable(data, variable, orig_scaler):
    scaler = StandardScaler()
    scaler.mean_ = orig_scaler[orig_scaler["variable"] == variable]["mean"].values[0]
    scaler.scale_ = orig_scaler[orig_scaler["variable"] == variable]["scale"].values[0]

    if isinstance(data, (pd.DataFrame, pd.Series)):
        inversed_data = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()
    else:
        inversed_data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    return inversed_data


def back_transform_variable(data, saturation_value=None):
    if saturation_value:
        back_transformed_data = saturation_value - np.exp(data)
    else:
        back_transformed_data = np.exp(data)
    return back_transformed_data


def inverse_back_transform(y_data, target_variable_name, scaler_df):
    saturation_values = {
        "hale": 1.1 * 72.7,
        "socialsupport": 1.1 * 0.985,
        "education": 1.1 * 19.69990921,
    }

    if isinstance(y_data, (pd.DataFrame, pd.Series)):
        y_data = y_data.values

    inversed_y_with_original_x = inverse_scale_variable(
        y_data, target_variable_name, scaler_df
    )
    back_transformed_predicted_y = back_transform_variable(
        inversed_y_with_original_x, saturation_values[target_variable_name]
    )

    return back_transformed_predicted_y


def predict_with_back_transform(
    model,
    target_variable_name,
    predictor_variable_names,
    back_transformed_data,
    transformed_data,
    scaler_df,
    interaction_terms=None,
    case=None,
    back_transform_y=False,
    include_dummies=False,
):
    data_original = back_transformed_data.copy()
    data_original = data_original.set_index(["geo", "TIME_PERIOD"])
    data_transformed = transformed_data.copy()
    dummies = pd.get_dummies(data_transformed["geo"], drop_first=True)
    data_transformed = data_transformed.set_index(["geo", "TIME_PERIOD"])

    first_non_energy_processed = (
        False  # Flag to track processing of the first non-energy variable
    )
    z_values = None
    z_values_energy = None
    for var in predictor_variable_names:
        if var == "energy":
            z_values_energy = data_transformed[var]
        else:
            if not first_non_energy_processed:
                z_values = data_transformed[var]
                # Apply specific logic only to the first non-energy variable
                if case == "LOW":
                    data_transformed[var] = data_transformed[var][
                        data_transformed[var] < data_transformed[var].median()
                    ].mean()
                elif case == "HIGH":
                    data_transformed[var] = data_transformed[var][
                        data_transformed[var] > data_transformed[var].median()
                    ].mean()
                else:
                    # set to mean
                    data_transformed[var] = data_transformed[var].mean()

                first_non_energy_processed = (
                    True  # Update flag after processing the first non-energy variable
                )
            else:
                # For all other variables, set to their overall mean
                data_transformed[var] = data_transformed[var].mean()

    exog_transformed = sm.add_constant(
        data_transformed[predictor_variable_names], has_constant="add"
    )

    if interaction_terms:
        for interaction_term in interaction_terms:
            interaction_term_name = f"{interaction_term[0]}_{interaction_term[1]}"
            exog_transformed[interaction_term_name] = (
                exog_transformed[interaction_term[0]]
                * exog_transformed[interaction_term[1]]
            )

    # If including dummies, ensure indices are compatible
    if include_dummies:
        exog_transformed_reset = exog_transformed.reset_index()
        exog_transformed_with_dummies = pd.concat(
            [exog_transformed_reset, dummies], axis=1
        )
        exog_transformed_with_dummies = exog_transformed_with_dummies.set_index(
            ["geo", "TIME_PERIOD"]
        )
        predicted_y = model.predict(exog_transformed_with_dummies)
    else:
        predicted_y = model.predict(exog_transformed)

    marginal_effects_df, marginal_effects_df_energy = None, None
    if len(predictor_variable_names) != 1:
        if isinstance(model, sm.regression.linear_model.RegressionResultsWrapper):
            vcov_matrix = model.cov_params()
        else:
            vcov_matrix = model.cov

        beta1 = model.params[predictor_variable_names[0]]
        beta2 = model.params[predictor_variable_names[1]]
        beta3 = model.params[f'{predictor_variable_names[0]}:{predictor_variable_names[1]}']

        beta1_var = vcov_matrix.loc[predictor_variable_names[0], predictor_variable_names[0]]
        beta2_var = vcov_matrix.loc[predictor_variable_names[1], predictor_variable_names[1]]
        beta3_var = vcov_matrix.loc[f'{predictor_variable_names[0]}:{predictor_variable_names[1]}', f'{predictor_variable_names[0]}:{predictor_variable_names[1]}']

        beta1_beta3_cov = vcov_matrix.loc[predictor_variable_names[0], f'{predictor_variable_names[0]}:{predictor_variable_names[1]}']
        beta2_beta3_cov = vcov_matrix.loc[predictor_variable_names[1], f'{predictor_variable_names[0]}:{predictor_variable_names[1]}']

        marginal_effects = []
        for z_value in z_values:
            marginal_effect = beta1 + z_value * beta3
            marginal_effects.append(marginal_effect)
        marginal_effects_energy = []
        for z_value_energy in z_values_energy:
            marginal_effect_energy = beta2 + z_value_energy * beta3
            marginal_effects_energy.append(marginal_effect_energy)

        marginal_effects_se = []
        for z_value in z_values:
            marginal_effect_var = beta1_var + (z_value ** 2) * beta3_var + 2 * z_value * beta1_beta3_cov
            marginal_effect_se = marginal_effect_var ** 0.5  # Square root of variance to get standard error
            marginal_effects_se.append(marginal_effect_se)
        marginal_effects_se_energy = []
        for z_value_energy in z_values_energy:
            marginal_effect_energy_var = beta2_var + (z_value_energy ** 2) * beta3_var + 2 * z_value_energy * beta2_beta3_cov
            marginal_effect_energy_se = marginal_effect_energy_var ** 0.5
            marginal_effects_se_energy.append(marginal_effect_energy_se)

        marginal_effects_df = pd.DataFrame(
            {
                "z_value": z_values,
                "marginal_effects": marginal_effects,
                "marginal_effects_se": marginal_effects_se,
            }
        )
        marginal_effects_df_energy = pd.DataFrame(
            {
                "z_value": z_values_energy,
                "marginal_effects": marginal_effects_energy,
                "marginal_effects_se": marginal_effects_se_energy,
            }
        )

    # Optionally back-transform the predicted y
    if back_transform_y:
        predicted_y_array = inverse_back_transform(
            predicted_y,
            target_variable_name,
            scaler_df,
        )
    else:
        predicted_y_array = predicted_y.values.squeeze()

    back_transformed_predicted_y_with_geo = pd.DataFrame(
        predicted_y_array, columns=[f"predicted_{target_variable_name}"]
    )
    back_transformed_predicted_y_with_geo[
        ["geo", "TIME_PERIOD"]
    ] = back_transformed_data[["geo", "TIME_PERIOD"]].reset_index(drop=True)

    data_original_with_y = data_original.reset_index().merge(
        back_transformed_predicted_y_with_geo, on=["geo", "TIME_PERIOD"], how="inner"
    )

    # add a column with first transformed x for plotting
    data_original_with_y[
        f"{predictor_variable_names[0]}_transformed"
    ] = data_transformed[predictor_variable_names[0]].reset_index(drop=True)

    # Return the dataframe sorted by the first predictor variable
    sorted_df_columns = (
        ["geo", "TIME_PERIOD"]
        + [predictor_variable_names[0]]
        + [f"predicted_{target_variable_name}"]
        + [f"{predictor_variable_names[0]}_transformed"]
    )
    sorted_df_with_predictors = data_original_with_y[sorted_df_columns]
    sorted_df_with_predictors = sorted_df_with_predictors.sort_values(
        by=predictor_variable_names[0]
    )

    return sorted_df_with_predictors, marginal_effects_df, marginal_effects_df_energy
