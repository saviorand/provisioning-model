from dataclasses import dataclass


@dataclass
class PeopleOutput:
    pass


@dataclass
class CapitalOutput:
    pass


@dataclass
class NatureOutput:
    pass


@dataclass
class EnergyOutput:
    pass


@dataclass
class NeedSatisfaction(PeopleOutput):
    nutrition: float  # from GoodLife, kilocalories per capita per day
    sanitation: float  # from GoodLife, percentage of population with access to improved sanitation
    income: float  # from GoodLife, percentage of population who earn above $1.90 per day
    life_satisfaction: float  # from GoodLife, [0-10] Cantril scale
    healthy_life_expectancy: float  # from GoodLife, years of healthy life
    education: float  # from GoodLife, percentage enrolment in secondary school
    social_support: float  # from GoodLife, percentage of population with friends or family they can depend on


@dataclass
class Accumulation(CapitalOutput):
    gdp_growth: float
    gni_growth: float
    gross_capital_formation: float
    corporate_profits_in_gdp: float
    patent_applications_per_capita: float
    trademark_applications_per_capita: float


@dataclass
class WasteAndPollution(NatureOutput):
    co2_emissions_per_capita: float
    phosphorus_discharge_per_capita: float
    nitrogen_discharge_per_capita: float
    blue_water_per_capita: float
    ehanpp_per_capita: float
    ecological_footprint_per_capita: float
    material_footprint_per_capita: float
