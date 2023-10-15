from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


@dataclass
class ComplexityAndInvestment:
    economic_complexity_index: float
    foreign_direct_investment_in_gdp: float
    rnd_investment_in_gdp: float


@dataclass
class Employment:
    informal_employment: float
    unemployment: float
    average_working_hours: float
    wage_worker_share: float
    self_employed_share: float
    unpaid_family_worker_share: float


@dataclass
class Trade:
    trade_penetration: float
    commercial_employment: float
    imports_in_gdp: float
    exports_in_gdp: float
    trade_diversification: float
    number_of_trade_partners: float
