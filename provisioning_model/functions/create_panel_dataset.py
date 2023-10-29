import pandas as pd
import seaborn as sns
from scipy import stats


def remove_cols_with_few_observations(df, threshold_country=75, threshold_year=75):
    """
    Find the best columns to keep in a DataFrame based on data precedence.

    Parameters:
        df (pd.DataFrame): Input DataFrame with observations for multiple variables,
                            countries, and years.
        threshold_country (float): Minimum percentage of countries that should have at least one
                                    non-NA/null data point in each column.
        threshold_year (float): Minimum percentage of years that should have at least one
                                 non-NA/null data point in each column.

    Returns:
        df (pd.DataFrame): DataFrame with only the columns that meet the criteria.
    """

    non_data_columns = [
        "Year",
        "Country.Code",
        "Country.Name",
    ]  # Columns that should not be considered in the analysis
    data_columns = [col for col in df.columns if col not in non_data_columns]

    # Number of unique countries and years
    num_countries = df["Country.Name"].nunique()
    num_years = df["Year"].nunique()

    # Initialize list to store columns that meet the criteria
    best_columns = []

    for col in data_columns:
        # For each column, filter rows where the data is not NA/null
        non_na_df = df[df[col].notna()]

        # Calculate the percentage of unique countries with data in this column
        unique_countries = non_na_df["Country.Name"].nunique()
        perc_countries = (unique_countries / num_countries) * 100

        # Calculate the percentage of unique years with data in this column
        unique_years = non_na_df["Year"].nunique()
        perc_years = (unique_years / num_years) * 100

        # If the column meets both country and year criteria, add it to best_columns
        if perc_countries >= threshold_country and perc_years >= threshold_year:
            best_columns.append(col)
    # keep year, country code and country name
    best_columns = non_data_columns + best_columns
    updated_df = df[best_columns]
    return updated_df


def create_outcome_df_with_metadata(df, outcome, index_offset, outcome_index_offset):
    summary = {}

    # Store information in the summary dictionary
    summary[outcome + "_df"] = df
    summary[outcome + "_indicators"] = df.iloc[
                                       :, index_offset: len(df.columns) - outcome_index_offset
                                       ]
    summary[outcome + "_outcome"] = df[outcome]
    summary[outcome + "_countries"] = df["Country.Name"].unique()
    summary[outcome + "_variables"] = df.columns[index_offset:]

    return summary


def remove_outliers_iqr(df, factor=1.5):
    """
    Remove outliers from a pandas DataFrame using the Interquartile Range (IQR).

    Parameters:
        df (pd.DataFrame): The DataFrame from which to remove outliers.
        factor (float): The factor to scale the IQR. Default is 1.5.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
    """
    # only apply to float64 columns
    df_float = df.select_dtypes(include=["float64"])

    # Calculate Q1, Q3, and IQR for each column
    Q1 = df_float.quantile(0.25)
    Q3 = df_float.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for the outliers
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Remove outliers
    df_filtered_float = df_float[~((df_float < lower_bound) | (df_float > upper_bound)).any(axis=1)]

    # return the original df with the outliers removed
    df_filtered = df[df_float.index.isin(df_filtered_float.index)]

    return df_filtered