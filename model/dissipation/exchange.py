from dataclasses import dataclass


@dataclass
class Trade:
    imports_in_gdp: float
    exports_in_gdp: float
    trade_diversification: float
    number_of_trade_partners: float
    foreign_direct_investment_in_gdp: float
    rnd_investment_in_gdp: float
