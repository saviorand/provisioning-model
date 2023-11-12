import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches


def get_country_data(df, outcome_df, realm_df, mat_flows_df, country, year):
    all_countries = df.loc[
        (df['Year'] == 2014)]
    all_countries_outcomes = outcome_df.loc[
        (outcome_df['Year'] == 2014)]
    all_countries_realms = realm_df.loc[
        (realm_df['Year'] == 2014)]
    all_countries_mat_flows = mat_flows_df.loc[
        (mat_flows_df['Year'] == 2014)]

    world_mean = all_countries.select_dtypes(include=[np.number]).mean()
    world_mean_outcomes = all_countries_outcomes.select_dtypes(include=[np.number]).mean()
    world_mean_realms = all_countries_realms.select_dtypes(include=[np.number]).mean()
    world_mean_mat_flows = all_countries_mat_flows.select_dtypes(include=[np.number]).mean()
    world_std = all_countries.select_dtypes(include=[np.number]).std()
    world_std_outcomes = all_countries_outcomes.select_dtypes(include=[np.number]).std()
    world_std_realms = all_countries_realms.select_dtypes(include=[np.number]).std()
    world_std_mat_flows = all_countries_mat_flows.select_dtypes(include=[np.number]).std()

    all_countries_energy_no_outliers = reject_outliers(all_countries["energy"])
    world_mean_energy = pd.Series()
    world_mean_energy["energy"] = all_countries_energy_no_outliers.mean()
    world_std_energy = pd.Series()
    world_std_energy["energy"] = all_countries_energy_no_outliers.std()

    data = df.loc[
        (df['Country.Name'] == country) & (df['Year'] == year)].iloc[0]
    outcomes = outcome_df.loc[
        (outcome_df['Country.Name'] == country) & (outcome_df['Year'] == year)].iloc[0]
    realms = realm_df.loc[
        (realm_df['Country.Name'] == country) & (realm_df['Year'] == year)].iloc[0]
    mat_flows = mat_flows_df.loc[
        (mat_flows_df['Country'] == country) & (mat_flows_df['Year'] == year)].iloc[0]

    normalized_data = normalize_series(data, world_mean, world_std,
                                       ["agriculture", "industry", "services", "grosscapital", "unemployed", "wealth"])
    normalized_outcomes = normalize_series(outcomes, world_mean_outcomes,
                                           world_std_outcomes, ["lifeexpectancy", "schoolenr"])
    normalized_realms = normalize_series(realms, world_mean_realms, world_std_realms,
                                         ["govconsum", "marketcap", "houseconsum"])
    normalized_mat_flows = normalize_series(mat_flows, world_mean_mat_flows,
                                            world_std_mat_flows, ["mfootprint"])
    normalized_energy = normalize_series(data, world_mean_energy, world_std_energy, ["energy"])

    agriculture, industry, services, grosscapital, unemployed, wealth = normalized_data.values()
    lifeexpectancy, education = normalized_outcomes.values()
    govconsum, marketcap, houseconsum = normalized_realms.values()
    energy = normalized_energy["energy"]
    mfootprint = normalized_mat_flows["mfootprint"]

    return {
        "agriculture": agriculture,
        "industry": industry,
        "services": services,
        "grosscapital": grosscapital,
        "unemployed": unemployed,
        "wealth": wealth,
        "lifeexpectancy": lifeexpectancy,
        "education": education,
        "govconsum": govconsum,
        "marketcap": marketcap,
        "houseconsum": houseconsum,
        "energy": energy,
        "mfootprint": mfootprint
    }


def reject_outliers(sr, iq_range=0.8):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1 - pcnt])
    iqr = qhigh - qlow
    return sr[(sr - median).abs() <= iqr]


def normalize_series(country_series, world_mean, world_std, categories):
    """
    Normalizes the values of a country's series with respect to the world mean and standard deviation.

    :param country_series: pandas Series containing the country's data.
    :param world_mean: pandas Series containing the world's mean for each category.
    :param world_std: pandas Series containing the world's standard deviation for each category.
    :return: A dictionary containing normalized values for each category.
    """

    normalized_data = {}

    for category in categories:
        if category == "agriculture":
            # Adding 1 to both country and world mean as per original calculation
            normalized_value = ((country_series[category] + 1) - (world_mean[category] + 1)) / world_std[category]
        else:
            normalized_value = (country_series[category] - world_mean[category]) / world_std[category]

        normalized_data[category] = normalized_value

    return normalized_data


def draw_arrow(ax1, x1, ax2, x2, y1=0.5, y2=0.5, arrowstyle="->", connectionstyle="arc3", color="black", linewidth=2,
               mutation_scale=20):
    """
    Draw an arrow between two circles located on different axes.

    Parameters:
    ax1: The source axes object.
    x1: The x-coordinate of the circle in the source axes.
    ax2: The destination axes object.
    x2: The x-coordinate of the circle in the destination axes.
    y1: The y-coordinate of the circle in the source axes.
    y2: The y-coordinate of the circle in the destination axes.
    arrowstyle: The style of the arrow.
    connectionstyle: The style of the connection between the two axes.
    color: The color of the arrow.
    linewidth: The width of the arrow line.
    mutation_scale: The size of the arrow head.
    """
    # Starting point in ax1
    xyA = (x1, y1)

    # Ending point in ax2
    xyB = (x2, y2)

    # Create a ConnectionPatch arrow
    arrow = patches.ConnectionPatch(
        xyA=xyA,
        xyB=xyB,
        coordsA=ax1.transData,
        coordsB=ax2.transData,
        arrowstyle=arrowstyle,
        connectionstyle=connectionstyle,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color
    )

    # Add the arrow to the figure
    ax1.figure.patches.append(arrow)
