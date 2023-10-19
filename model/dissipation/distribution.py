from dataclasses import dataclass


# how output is distributed, incl. to third parties
# what part is consumed by whom, what is reinvested into what, what is accumulated where
# what part of output goes to waste


@dataclass
class Trade:
    imports_in_gdp: float
    exports_in_gdp: float
    trade_diversification: float
    number_of_trade_partners: float
    foreign_direct_investment_in_gdp: float
    rnd_investment_in_gdp: float


@dataclass
class Waste:
    volume: float
    structure: float
    recycling_rate: float


@dataclass
class EconomicInequality:
    gini_index: float
    atkinson_index: float
    top_1_percent_income_share: float
    top_1_percent_wealth_share: float


@dataclass
class ProfitStructure:
    profits_of_corporations_in_gdp: float
    profits_of_state_in_gdp: float


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
