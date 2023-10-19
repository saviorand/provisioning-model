from dataclasses import dataclass


@dataclass
class PeopleOutput:
    pass


@dataclass
class CapitalOutput:
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
