import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from adjustText import adjust_text


def draw_heatmap(df, time_col, country_col, energy_col, time_ticks):
    fig = px.density_heatmap(
        df,
        x=country_col,
        y=time_col,
        z=energy_col,
        color_continuous_scale="Viridis",
        title="Energy Usage by Country Over Time",
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig.update_yaxes(tickvals=time_ticks)
    return fig


def create_scatter_plots_twodim_with_fit(df, time_col, cols, outcome_col):
    years = sorted(df[time_col].unique())  # Sort years in ascending order
    colors = ["red", "green", "blue", "purple"]
    line_color = "black"  # Distinct color for the regression line
    subplot_titles = []
    for year in years:
        year_title = [f"Year: {year}, {cols[0]}"] + [f"{var}" for var in cols[1:]]
        subplot_titles.extend(year_title)
    fig = make_subplots(
        rows=len(years),
        cols=len(cols),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.09,
    )
    for i, year in enumerate(years):
        for j, var in enumerate(cols):
            df_year = df[df[time_col] == year]
            x = df_year[var]
            y = df_year[outcome_col]
            valid_indices = ~np.isnan(x) & ~np.isnan(y)
            x = x[valid_indices]
            y = y[valid_indices]
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="markers", name=var, marker_color=colors[j]),
                row=i + 1,
                col=j + 1,
            )
            if len(x) > 1:
                try:
                    slope, intercept, _, _, _ = linregress(x, y)
                    line_x = np.linspace(min(x), max(x), 100)
                    line_y = slope * line_x + intercept
                    fig.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode="lines",
                            name=f"Fit {var}",
                            line=dict(color=line_color),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                except Exception as e:
                    print(
                        f"Error in linear regression calculation for {var} in {year}: {e}"
                    )
    fig.update_layout(
        height=800,
        width=1000,
        title_text=f"y-axis: {outcome_col}, x-axis: foundational type % share",
        showlegend=False,
    )
    return fig


def create_scatter_plots_twodim(df, time_col, cols, outcome_col):
    years = df[time_col].unique()
    colors = ["red", "green", "blue", "purple"]
    fig = make_subplots(
        rows=4,
        cols=len(years),
        subplot_titles=[f"{year}" for year in years],
        vertical_spacing=0.09,
    )
    fig = make_subplots(
        rows=4, cols=len(years), subplot_titles=[f"Year: {year}" for year in years]
    )
    for i, var in enumerate(cols):
        for j, year in enumerate(years):
            df_year = df[df["TIME_PERIOD"] == year]
            fig.add_trace(
                go.Scatter(
                    x=df_year[var],
                    y=df_year[outcome_col],
                    mode="markers",
                    name=var,
                    marker_color=colors[i],
                ),
                row=i + 1,
                col=j + 1,
            )
    fig.update_layout(
        height=800, width=1000, title_text="Scatter Plots Grid", showlegend=False
    )
    return fig


def get_extreme_points(x, y, lower_percentile, upper_percentile):
    # Extreme points
    low_x, high_x = np.percentile(x, [lower_percentile, upper_percentile])
    low_y, high_y = np.percentile(y, [lower_percentile, upper_percentile])
    extreme_indices = np.where(
        (x <= low_x) | (x >= high_x) | (y <= low_y) | (y >= high_y)
    )[0]
    return extreme_indices


def get_middle_points(x, y, n_middle):
    median_x = np.median(x)
    median_y = np.median(y)
    distances = np.sqrt((x - median_x) ** 2 + (y - median_y) ** 2)
    middle_indices = np.argsort(distances)[:n_middle]
    return middle_indices


def create_scatter_plots_grid(
    df,
    time_col,
    cols,
    time_periods,
    titles=[],
    countries_to_annotate=[],
    save_path=None,
    height=600,
    width=800,
    rows=2,
):
    # if len(cols) > 4:
    #     raise ValueError("Maximum of 4 columns can be plotted in a 2x2 grid.")

    fig = make_subplots(rows=rows, cols=2, subplot_titles=cols[:4])

    # Manually select indices of points to annotate
    # For example, [0, -1] will annotate the first and last points in your dataset
    indices_to_annotate = []
    for country in countries_to_annotate:
        country_index = df[
            (df["geo"] == country["geo"]) & (df[time_col] == country[time_col])
        ].index
        if len(country_index) > 0:
            indices_to_annotate.append(
                {"country": country_index[0], "show": country["show"]}
            )
        else:
            print(f"Country {country['geo']} not found in the dataset")

    for i, col in enumerate(cols[:4]):
        row_num = i // 2 + 1
        col_num = i % 2 + 1
        x = df[time_col]
        y = df[col]

        # Add scatter trace for markers
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=col,
                hovertext=df["geo"],
            ),
            row=row_num,
            col=col_num,
        )
        for idx in indices_to_annotate:
            if idx["show"] == col:
                country_index = idx["country"]
                fig.add_annotation(
                    x=df.iloc[country_index][
                        time_col
                    ],  # Use country_index to access the specific row
                    y=df.iloc[country_index][
                        col
                    ],  # Use country_index to access the specific row
                    text=f"{df.iloc[country_index]['geo']} {df.iloc[country_index][time_col]}: {float(df.iloc[country_index][col]):.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xref="x",
                    yref="y",
                    ax=20,  # Adjust these for arrow positioning
                    ay=-30,  # Adjust these for arrow positioning
                    row=row_num,
                    col=col_num,
                )

        fig.update_xaxes(tickvals=time_periods, row=row_num, col=col_num)

    fig.update_layout(height=height, width=width, showlegend=False)
    if save_path:
        # save as png
        fig.write_image(save_path)
    fig.show()


def plot_overlayed_histograms(df, cols, title):
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Histogram(x=df[col], name=col))
    fig.update_layout(barmode="overlay", title_text=title)
    fig.update_traces(opacity=0.55, hoverinfo="all")
    fig.show()


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
        sub_df = df[["geo", "TIME_PERIOD", col]].dropna()

        observations = len(sub_df)
        year_start = sub_df["TIME_PERIOD"].min() if not sub_df.empty else None
        year_end = sub_df["TIME_PERIOD"].max() if not sub_df.empty else None
        num_countries = sub_df["geo"].nunique() if not sub_df.empty else 0
        num_years = sub_df["TIME_PERIOD"].nunique() if not sub_df.empty else 0

        available_countries = set(sub_df["geo"])
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


def plot_outliers(df):
    """
    Create a boxplot of a given dataframe.

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

    # Determine the order of the columns based on hierarchical clustering
    corr_condensed = -corr.abs()  # Convert correlation to distance
    linkage = sns.clustermap(
        corr_condensed,
        method="average",
        metric="euclidean",
        figsize=(1, 1),
        cbar_pos=None,
    ).dendrogram_col.linkage
    plt.close()  # Close the clustermap plot
    order = sns.clustermap(
        corr, row_linkage=linkage, col_linkage=linkage, figsize=(1, 1), cbar_pos=None
    ).dendrogram_col.reordered_ind
    plt.close()  # Close the clustermap plot
    sorted_corr = corr.iloc[order, order]

    # Create a mask to hide the upper triangle of the correlation matrix (since it's symmetric)
    mask = np.triu(np.ones_like(sorted_corr, dtype=bool))

    # Plot the heatmap with the mask
    plt.figure(figsize=(12, 8))
    sns.heatmap(sorted_corr, mask=mask, cmap="coolwarm", annot=annot, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Identify and print names of columns with high correlation
    high_corr_pairs = []
    for i in range(len(sorted_corr.columns)):
        for j in range(i):
            if abs(sorted_corr.iloc[i, j]) > threshold:
                high_corr_pairs.append(
                    (
                        sorted_corr.columns[i],
                        sorted_corr.columns[j],
                        sorted_corr.iloc[i, j],
                    )
                )
                # print(f"{sorted_corr.columns[i]} and {sorted_corr.columns[j]}: {sorted_corr.iloc[i, j]:.2f}")

    return high_corr_pairs


def plot_histograms(df, cols, bins=50):
    """
    Plots a histogram for each column in a given list of columns on a single figure.

    Parameters:
        df (pd.DataFrame): The DataFrame for which to plot the histograms.
        cols (list): The list of columns for which to plot the histograms.
        bins (int): The number of bins to use for the histogram.

    Returns:
        None
    """

    # Number of rows and columns for subplot grid
    n = len(cols)
    n_cols = 2  # For instance, you can adjust this value as per your preference.
    n_rows = int(n / n_cols) + (
        n % n_cols
    )  # Calculate the number of rows required based on the number of columns

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 5 * n_rows)
    )  # Adjust size based on the number of plots

    # Flatten the axes array if there's only one row
    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)

    for idx, col in enumerate(cols):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.hist(df[col].dropna(), bins=bins)  # Drop NaN values before plotting
        ax.set_title(f"Histogram of {col}")

    # Remove any unused subplots
    for j in range(n, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()
