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
    food: float
    water: float
    shelter: float


@dataclass
class Accumulation(CapitalOutput):
    gdp_growth: float
    gni_growth: float
    gross_capital_formation: float
    corporate_profits_in_gdp: float
    patent_applications_per_capita: float
    trademark_applications_per_capita: float


@dataclass
class HumanDevelopment(PeopleOutput):
    life_expectancy: float
    education_index: float
    health_expenditure_in_gdp: float
    education_expenditure_in_gdp: float
    internet_users_per_capita: float
    mobile_subscriptions_per_capita: float
    press_freedom_index: float
    happiness_index: float


@dataclass
class ImpactOnEcosystems(NatureOutput):
    pass


@dataclass
class WasteAndPollution(NatureOutput):
    volume: float
    structure: float
    recycling_rate: float