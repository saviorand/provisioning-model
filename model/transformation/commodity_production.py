import reflex as rx


class ProductionStructure(rx.Model, table=True):
    economic_complexity_index: float
    agriculture_share: float
    industry_share: float
    services_share: float
    financial_services_share: float
    manufacturing_share: float
    mining_share: float
    construction_share: float


class InvestmentStructure(rx.Model, table=True):
    foreign_direct_investment_in_gdp: float
    rnd_investment_in_gdp: float
    infrastructure_investment_in_gdp: float
    agriculture_investment_in_gdp: float
    industry_investment_in_gdp: float
    services_investment_in_gdp: float


class EmploymentStructure(rx.Model, table=True):
    agriculture_employment: float
    industry_employment: float
    services_employment: float
    informal_employment: float
    unemployment: float
    average_working_hours: float
    wage_worker_share: float
    self_employed_share: float
    unpaid_family_worker_share: float


class InternalTradeStructure(rx.Model, table=True):
    trade_penetration: float
    commercial_employment: float


class ForeignTradeStructure(rx.Model, table=True):
    imports_in_gdp: float
    exports_in_gdp: float
    trade_diversification: float
    number_of_trade_partners: float


class CommodityProduction(rx.Model):
    production_structure: ProductionStructure
    investment_structure: InvestmentStructure
    employment_structure: EmploymentStructure
    internal_trade_structure: InternalTradeStructure
    foreign_trade_structure: ForeignTradeStructure
