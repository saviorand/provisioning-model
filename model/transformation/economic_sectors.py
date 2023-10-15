from dataclasses import dataclass


@dataclass
class Sector:
    value_added_share: float
    employment_share: float
    investment_share: float
    export_share: float
    import_share: float


@dataclass
class Agriculture(Sector):
    cereals_fruit_and_vegetables: Sector
    livestock_and_fish: Sector
    forestry: Sector


@dataclass
class Industry:
    light_industry: Sector
    heavy_industry: Sector
    construction: Sector
    mining: Sector


@dataclass
class Services:
    infrastructure: Sector
    care_health_education: Sector
    other_services: Sector


@dataclass
class FinanceAndTrade:
    retail_and_commerce: Sector
    fire: Sector


@dataclass
class Sectors:
    agriculture: Agriculture
    industry: Industry
    services: Services
    finance_and_trade: FinanceAndTrade
