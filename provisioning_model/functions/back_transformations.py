import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def inverse_scale_variable(data, variable, orig_scaler):
    scaler = StandardScaler()
    scaler.mean_ = orig_scaler[orig_scaler["variable"] == variable]["mean"].values[0]
    scaler.scale_ = orig_scaler[orig_scaler["variable"] == variable]["scale"].values[0]
    if isinstance(data, (pd.DataFrame, pd.Series)):
        # Use .values or .to_numpy() to get a numpy array from the DataFrame or Series
        # Then reshape it for the scaler
        inversed_data = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()
    else:
        # Assuming 'data' is already a numpy array here
        inversed_data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    return inversed_data


def back_transform_variable(data, saturation_value=None):
    if saturation_value:
        back_transformed_data = saturation_value - np.exp(data)
    else:
        back_transformed_data = np.exp(data)
    return back_transformed_data


def inverse_back_transform(y_data, target_variable_name, scaler_df):
    saturation_values = {"hale": 1.1 * 72.7}

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
):
    data_original = back_transformed_data.copy()
    data_original = data_original.set_index(["geo", "TIME_PERIOD"])
    data_transformed = transformed_data.copy()
    data_transformed = data_transformed.set_index(["geo", "TIME_PERIOD"])

    for var in predictor_variable_names:
        if var != "energy":
            if case == "LOW":
                # set to mean of the below 50% of the data
                data_transformed[var] = data_transformed[var][
                    data_transformed[var] < data_transformed[var].median()
                ].mean()
            elif case == "HIGH":
                # set to mean of the above 50% of the data
                data_transformed[var] = data_transformed[var][
                    data_transformed[var] > data_transformed[var].median()
                ].mean()
            else:
                # set to mean
                data_transformed[var] = data_transformed[var].mean()

    exog_transformed = sm.add_constant(
        data_transformed[predictor_variable_names], has_constant="add"
    )

    if interaction_terms:
        interaction_term_name = (
            f"{predictor_variable_names[0]}_{predictor_variable_names[1]}"
        )
        exog_transformed[interaction_term_name] = (
            exog_transformed[predictor_variable_names[0]]
            * exog_transformed[predictor_variable_names[1]]
        )

    predicted_y = model.predict(exog_transformed)

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
        + predictor_variable_names
        + [f"predicted_{target_variable_name}"]
        + [f"{predictor_variable_names[0]}_transformed"]
    )
    sorted_df_with_predictors = data_original_with_y[sorted_df_columns]
    sorted_df_with_predictors = sorted_df_with_predictors.sort_values(
        by=predictor_variable_names[0]
    )

    return sorted_df_with_predictors
