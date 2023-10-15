from dataclasses import dataclass


@dataclass
class DemographicsAndUrbanization:
    population: float
    population_growth_rate: float
    ethnic_diversity: float
    urbanization_rate: float
    urbanization_level: float
    regional_income_disparity: float


@dataclass
class NaturalResources:
    land_area: float
    forest_area: float
    fossil_fuel_reserves: float
    rare_minerals_per_cap: float
    clean_water_per_cap: float
    climate_index: float


@dataclass
class CapitalEndowment:
    gdp_per_capita: float
    per_adult_national_wealth: float
    wealth_per_capita: float
    wealth_income_ratio: float
    capital_per_worker: float
    capital_per_capita: float
    patents_per_capita: float
    trademarks_per_capita: float


@dataclass
class InitialConditions:
    natural_resources: NaturalResources
    capital_endowment: CapitalEndowment
    demographics_and_urbanization: DemographicsAndUrbanization
