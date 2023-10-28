from dataclasses import dataclass


# goal-setting, how to decide what to produce, how priorities and prices are set, e.g. market, state, monopolies

@dataclass
class Government:
    government_effectiveness_index: float
    government_spending_in_gdp: float
    one_party_system: bool
    military_rule: bool
    leader_ideology: bool
    corruption_perception_index: float
    protests_and_riots_per_cap: float


@dataclass
class CorporatePower:
    market_concentration: float
    corporate_tax_rate: float
    lobbying_expenditure_in_gdp: float


@dataclass
class Commercialization:
    trade_penetration: float
    commercial_employment: float


@dataclass
class LegalSystem:
    pass
