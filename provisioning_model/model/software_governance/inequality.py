from dataclasses import dataclass


@dataclass
class EconomicInequality:
    gini_index: float
    atkinson_index: float
    top_1_percent_income_share: float
    top_1_percent_wealth_share: float


@dataclass
class GenderInequality:
    women_parliament_share: float
    women_minister_share: float
    women_management_share: float
    women_labour_income_share: float
    gender_pay_gap: float


@dataclass
class EthnicInequality:
    ethnic_inequality_index: float
    ethnic_pay_gap: float
    minority_parliament_share: float
    minority_minister_share: float
    minority_management_share: float
    minority_labour_income_share: float


@dataclass
class CrimeAndDespair:
    suicide_rate: float
    homicides_per_capita: float
    property_crime_per_capita: float
    incarceration_rate: float
