import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_best_subset_results(model_coef, predictor_df):
    """
    Plots non-zero coefficients of the model against corresponding variable names.

    Parameters:
        model_coef (array-like): Coefficients of the model.
        predictor_df (pd.DataFrame): DataFrame containing the predictor variables.

    Returns:
        None
    """
    # Extract column names and coefficients from the DataFrame
    column_names = predictor_df.columns

    # Find non-zero coefficients
    non_zero_indices = np.nonzero(model_coef)[0]

    # Extract non-zero coefficients and corresponding labels
    non_zero_coef = model_coef[non_zero_indices]
    non_zero_labels = column_names[non_zero_indices]

    # Create a DataFrame to sort by absolute coefficient values
    df = pd.DataFrame({'Variable': non_zero_labels, 'Coefficient': non_zero_coef})
    df = df.sort_values(by='Coefficient', key=abs, ascending=True)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.barh(df['Variable'], df['Coefficient'], color='skyblue')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Variables')
    plt.title('Non-zero Coefficients')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add text annotations to the bars
    for index, value in enumerate(df['Coefficient']):
        plt.text(value, index, f'{value:.4f}')

    plt.show()

# Example usage:
# plot_best_subset_results(model.coef_, indicators_lifeexp_over_energy_imputed_df)
