from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


@dataclass
class EconomicGrowth():
    gdp_growth: float
    gni_growth: float


@dataclass
class CapitalFormation():
    gross_capital_formation: float
    private_gross_capital_formation: float
    gross_fixed_capital_formation: float
    savings_in_gdp: float
    material_stock_growth_in_gdp: float
    annual_rate_of_profit_of_corporations: float


@dataclass
class CapitalEndowment():
    gdp_per_capita: float
    per_adult_national_wealth: float
    wealth_per_capita: float
    wealth_income_ratio: float
    capital_per_worker: float
    capital_per_capita: float


@dataclass
class ProfitStructure():
    profits_of_corporations_in_gdp: float
    profits_of_state_in_gdp: float
    share_of_profits_in_industry: float
    share_of_profits_in_agriculture: float
    share_of_profits_in_services: float
    share_of_profits_in_financial_services: float


@dataclass
class IntellectualProperty():
    patents_per_capita: float
    trademarks_per_capita: float
    patent_applications_per_capita: float
    trademark_applications_per_capita: float


@dataclass
class Accumulation():
    economic_growth: EconomicGrowth
    capital_formation: CapitalFormation
    capital_endowment: CapitalEndowment
    profit_structure: ProfitStructure
    intellectual_property: IntellectualProperty
