from dataclasses import dataclass


# make distinction between productive stock, housing, intellectual property etc
@dataclass
class CapitalBase:
    gdp_per_capita: float
    per_adult_national_wealth: float
    wealth_per_capita: float
    wealth_income_ratio: float
    capital_per_worker: float
    capital_per_capita: float
    patents_per_capita: float
    trademarks_per_capita: float
