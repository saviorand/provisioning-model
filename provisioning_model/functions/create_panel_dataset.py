import pandas as pd
import numpy as np
import pandas as pd
import re

def custom_log_transform(value, min_value, max_value):
    if min_value >= 0 and max_value <= 1:
        # Case: All values are between 0 and 1
        offset = 1
    elif min_value < 0:
        # Case: There are negative values
        offset = abs(min_value) + 1
    else:
        # Case: Positive values, no offset needed
        offset = 0

    return np.log(value + offset)


def remove_outliers(outliers_to_remove, df, countries_to_remove=None):
    df_no_outliers = df.copy()
    for outlier in outliers_to_remove:
        df_no_outliers = df_no_outliers[~((df_no_outliers['geo'] == outlier['geo']) & (df_no_outliers['TIME_PERIOD'] == outlier['TIME_PERIOD']))]
    for country in countries_to_remove:
        df_no_outliers = df_no_outliers[df_no_outliers['geo'] != country]
    
    # Print the number of rows before and after outlier removal
    print('Number of rows in df: {}'.format(len(df)))
    print('Number of rows in df_no_outliers: {}'.format(len(df_no_outliers)))
    
    # Return the modified dataframes
    return df_no_outliers

def clean_and_transform_wdi(df, value_name):
    """
    Apply the transformations to the given dataframe and return the cleaned dataframe.
    The dataframe will be reshaped to have columns "Year", "Country.Code", "Country.Name", and "value_name".
    """
    # Extract columns that match the pattern 'XXXX [YRXXXX]' for year
    year_columns = [col for col in df.columns if re.match(r"\d{4} \[YR\d{4}\]", col)]

    df_melted = df.melt(id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
                        value_vars=year_columns,
                        var_name="Year",
                        value_name=value_name)


    # Extract the year from the "Year" column
    df_melted["Year"] = df_melted["Year"].str.extract("(\d{4})").astype(int)

    # Drop unnecessary columns
    df_melted.drop(columns=["Series Code", "Series Name"], inplace=True)

    # Rename the columns
    df_melted.rename(columns={
        "Country Name": "Country.Name",
        "Country Code": "Country.Code",
    }, inplace=True)

    # Sort the dataframe by country_code and then year
    df_melted.sort_values(by=["Country.Name", "Year"], inplace=True)

    # Reorder columns
    df_melted = df_melted[["Year", "Country.Code", "Country.Name", value_name]]

    # Remove rows where country code is NA
    df_melted = df_melted.dropna(subset=["Country.Code"])

    # Replace ".." with NaN
    df_melted[value_name].replace("..", pd.NA, inplace=True)

    # Check if the column's dtype is 'object'
    if df_melted[value_name].dtype == 'object':
        # Convert columns with object dtype that contain numeric data to numeric columns
        num_col = pd.to_numeric(df_melted[value_name], errors='coerce')
        # Check if the column isn't all NaN (which would indicate it was a non-numeric column)
        if not num_col.isna().all():
            df_melted[value_name] = num_col

    return df_melted

def clean_and_transform_wdi_outcomes(df, value_name):
    """
    Apply the transformations to the given dataframe and return the cleaned dataframe.
    The dataframe will be reshaped to have columns "Year", "Country.Code", "Country.Name", and "value_name".
    """
    # Extract columns that match the pattern 'XXXX [YRXXXX]' for year
    year_columns = [col for col in df.columns if re.match(r"\d{4} \[YR\d{4}\]", col)]

    df_melted = df.melt(id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
                        value_vars=year_columns,
                        var_name="Year",
                        value_name=value_name)

    # Extract the year from the "Year" column
    df_melted["Year"] = df_melted["Year"].str.extract("(\d{4})").astype(int)

    # Drop unnecessary columns
    df_melted.drop(columns=["Series Code", "Series Name"], inplace=True)

    # Rename the columns
    df_melted.rename(columns={
        "Country Name": "Country.Name",
        "Country Code": "Country.Code",
    }, inplace=True)

    # Sort the dataframe by country_code and then year
    df_melted.sort_values(by=["Country.Name", "Year"], inplace=True)

    # Reorder columns
    df_melted = df_melted[["Year", "Country.Code", "Country.Name", value_name]]

    # Remove rows where country code is NA
    df_melted = df_melted.dropna(subset=["Country.Code"])

    # Replace ".." with NaN
    df_melted[value_name].replace("..", pd.NA, inplace=True)

    # Check if the column's dtype is 'object'
    if df_melted[value_name].dtype == 'object':
        # Convert columns with object dtype that contain numeric data to numeric columns
        num_col = pd.to_numeric(df_melted[value_name], errors='coerce')
        # Check if the column isn't all NaN (which would indicate it was a non-numeric column)
        if not num_col.isna().all():
            df_melted[value_name] = num_col

    return df_melted
def calculate_shares(df, value_col):
    """
    Calculate the share of each category in a value column.
    :param df: the input DataFrame
    :param value_col: the column containing the values to calculate the share of
    :return: a DataFrame with the share of each category in a value column
    """
    # Group by 'geo', 'TIME_PERIOD', and 'final_foundational' and sum the specified value column
    grouped = df.groupby(['geo', 'TIME_PERIOD', 'final_foundational'])[value_col].sum().reset_index()

    # Calculate total value for each country-year combination
    total_values = df.groupby(['geo', 'TIME_PERIOD'])[value_col].sum().reset_index()
    total_values.rename(columns={value_col: 'total_value'}, inplace=True)

    # Merge and calculate share
    merged = pd.merge(grouped, total_values, on=['geo', 'TIME_PERIOD'])
    merged['share'] = merged[value_col] / merged['total_value']

    # Pivot the table to get categories as columns
    pivot_table = merged.pivot_table(index=['geo', 'TIME_PERIOD'], columns='final_foundational', values='share', fill_value=0).reset_index()

    return pivot_table

def standardize_data(input_data) -> np.ndarray:
    """
    Standardize data by subtracting the mean and dividing by the standard deviation.
    :param input_data: a numpy array of data
    :return: numpy array of standardized data
    """
    # Calculate mean and standard deviation, ignoring NaNs
    mean = np.nanmean(input_data)
    std_dev = np.nanstd(input_data)

    # Standardize data
    output_data = (input_data - mean) / std_dev

    return output_data

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
