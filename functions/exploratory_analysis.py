import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_summary_df(
    df: pd.DataFrame, countries: list, variables: list
) -> pd.DataFrame:
    """
    Create a summary dataframe for a given dataframe describing the number of observations, the first and last year, the number of countries and the number of years for each variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to summarize.
    countries : list
        The list of countries to consider.
    variables : list
        The list of variables to consider.

    Returns
    -------
    pandas.DataFrame
        The summary dataframe.
    """
    summary_df = pd.DataFrame(
        columns=[
            "observations",
            "year_start",
            "year_end",
            "num_countries",
            "num_years",
            "missing_countries",
        ]
    )

    for col in variables:
        sub_df = df[["Country.Name", "Year", col]].dropna()

        observations = len(sub_df)
        year_start = sub_df["Year"].min() if not sub_df.empty else None
        year_end = sub_df["Year"].max() if not sub_df.empty else None
        num_countries = sub_df["Country.Name"].nunique() if not sub_df.empty else 0
        num_years = sub_df["Year"].nunique() if not sub_df.empty else 0

        available_countries = set(sub_df["Country.Name"])
        missing_countries = list(set(countries) - available_countries)

        summary_df.loc[col] = [
            observations,
            year_start,
            year_end,
            num_countries,
            num_years,
            missing_countries,
        ]

    return summary_df


def plot_summary_variable(summary_df: pd.DataFrame, summary_variable) -> None:
    """
    Create a bar plot of a given summary variable. Y axis is the summary variable and X axis is the country names.

    Parameters
    ----------
    summary_df : DataFrame
        The DataFrame containing the data.
    summary_variable : str
        The summary variable to plot.
    """
    # Sort the summary_df based on the variable
    sorted_summary_df = summary_df.sort_values(by=summary_variable, ascending=False)

    # Create the plot
    plt.figure(figsize=(15, 6))
    sns.barplot(x=sorted_summary_df.index, y=sorted_summary_df[summary_variable])

    # Customize the plot
    plt.xlabel("Variables")
    plt.ylabel(summary_variable)
    plt.title("Summary of {}".format(summary_variable))
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()


def plot_outliers(df, variable):
    """
    Create a boxplot of a given variable for a given list of countries.

    Parameters:
        df (pd.DataFrame): Input DataFrame with observations for multiple variables,
                            countries, and years.
    """
    sns.boxplot(data=df, orient="h", palette="Set2")
    plt.show()


def plot_correlation_matrix(df, threshold=0.8, annot=False):
    """
    Plots a correlation matrix for a given pandas DataFrame and prints names of columns with high correlation.

    Parameters:
        df (pd.DataFrame): The DataFrame for which to plot the correlation matrix.
        threshold (float): The absolute correlation coefficient value above which to report high correlations.
        annot (bool): Whether to annotate each cell with the numeric value of the correlation.

    Returns:
        list: A list of tuple containing pairs of highly correlated columns along with their correlation coefficient.
    """
    # Compute the correlation matrix
    corr = df.corr()

    # Create a mask to hide the upper triangle of the correlation matrix (since it's symmetric)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot the heatmap with the mask
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=annot, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Identify and print names of columns with high correlation
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                high_corr_pairs.append(
                    (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                )
                print(f"{corr.columns[i]} and {corr.columns[j]}: {corr.iloc[i, j]:.2f}")

    return high_corr_pairs


def plot_histograms(df, cols, bins=50):
    """
    Plots a histogram for each column in a given list of columns.

    Parameters:
        df (pd.DataFrame): The DataFrame for which to plot the histograms.
        cols (list): The list of columns for which to plot the histograms.
        bins (int): The number of bins to use for the histogram.

    Returns:
        None
    """
    for col in cols:
        plt.figure(figsize=(8, 4))
        plt.hist(df[col], bins=bins)
        plt.title(f"Histogram of {col}")
        plt.show()
