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


def create_outcome_summaries(
    df,
    outcome_variables,
    filter_year=None,
    index_offset=3,
    outcome_index_offset=17,
    remove_cols_func=None,
    remove_outliers_func=None,
):
    """
    Create summaries for each outcome variable in a list.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        outcome_variables (list): List of outcome variables to summarize.
        filter_year (int): The year to filter the DataFrame by for cross-sectional analysis. Default is None.
        index_offset (int): The number of columns to skip from the beginning to exclude index variables
        outcome_index_offset (int): The number of columns to take from the last column to exclude outcomes
        remove_cols_func (function): Function to remove columns with few observations. Default is None.
        remove_outliers_func (function): Function to remove outliers. Default is None.

    Returns:
        list: A list of dictionaries containing summaries for each outcome variable.
    """

    outcome_summaries = []

    # Filter the DataFrame by the specified year if provided
    if filter_year is not None:
        df = df[df["Year"] == filter_year]

    for outcome in outcome_variables:
        outcome_summary = {}

        # Drop rows where the outcome variable is NaN
        outcome_df = df.dropna(subset=[outcome])

        # Remove columns with few observations if a function is provided
        if remove_cols_func:
            outcome_df_no_few_obs = remove_cols_func(outcome_df)
        else:
            outcome_df_no_few_obs = outcome_df

        # Remove outliers if a function is provided
        if remove_outliers_func:
            outcome_df_no_outliers = remove_outliers_func(outcome_df_no_few_obs)
        else:
            outcome_df_no_outliers = outcome_df_no_few_obs

        # Store various pieces of information in the summary dictionary
        outcome_summary[outcome + "_df"] = outcome_df_no_outliers
        outcome_summary[outcome + "_indicators"] = outcome_df_no_outliers.iloc[
            :, index_offset : len(outcome_df_no_outliers.columns) - outcome_index_offset
        ]
        outcome_summary[outcome + "_outcome"] = outcome_df_no_outliers[outcome]
        outcome_summary[outcome + "_countries"] = outcome_df_no_outliers[
            "Country.Name"
        ].unique()
        outcome_summary[outcome + "_variables"] = outcome_df_no_outliers.columns[
            index_offset:
        ]

        outcome_summaries.append(outcome_summary)

    return outcome_summaries


def remove_outliers_iqr(df, factor=1.5):
    """
    Remove outliers from a pandas DataFrame using the Interquartile Range (IQR).

    Parameters:
        df (pd.DataFrame): The DataFrame from which to remove outliers.
        factor (float): The factor to scale the IQR. Default is 1.5.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
    """

    # Calculate Q1, Q3, and IQR for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for the outliers
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    # Remove outliers
    df_filtered = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

    return df_filtered
