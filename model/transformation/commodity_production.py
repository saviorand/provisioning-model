from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


@dataclass
class ProductionStructure:
    economic_complexity_index: float
    agriculture_share: float
    industry_share: float
    services_share: float
    financial_services_share: float
    manufacturing_share: float
    mining_share: float
    construction_share: float


@dataclass
class InvestmentStructure:
    foreign_direct_investment_in_gdp: float
    rnd_investment_in_gdp: float
    infrastructure_investment_in_gdp: float
    agriculture_investment_in_gdp: float
    industry_investment_in_gdp: float
    services_investment_in_gdp: float


@dataclass
class EmploymentStructure:
    agriculture_employment: float
    industry_employment: float
    services_employment: float
    informal_employment: float
    unemployment: float
    average_working_hours: float
    wage_worker_share: float
    self_employed_share: float
    unpaid_family_worker_share: float


@dataclass
class InternalTradeStructure:
    trade_penetration: float
    commercial_employment: float


@dataclass
class ForeignTradeStructure:
    imports_in_gdp: float
    exports_in_gdp: float
    trade_diversification: float
    number_of_trade_partners: float


@dataclass
class CommodityProduction:
    production_structure: ProductionStructure
    investment_structure: InvestmentStructure
    employment_structure: EmploymentStructure
    internal_trade_structure: InternalTradeStructure
    foreign_trade_structure: ForeignTradeStructure
