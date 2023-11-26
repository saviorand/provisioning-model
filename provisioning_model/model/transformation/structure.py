from dataclasses import dataclass


# industries/kinds of commodities and services produced, complexity/diversity
# goal-setting, how to decide what to produce, how priorities and prices are set, e.g. market, state, monopolies

@dataclass
class IndustralSector:
    pass


@dataclass
class Agriculture(IndustralSector):
    pass


@dataclass
class Industry(IndustralSector):
    pass


@dataclass
class Services(IndustralSector):
    pass


@dataclass
class FoundationalSector:
    pass


@dataclass
class FoundationalEconomy(FoundationalSector):
    pass


@dataclass
class OverlookedEconomy(FoundationalSector):
    pass


@dataclass
class CoreEconomy(FoundationalSector):
    pass


@dataclass
class CompetitiveEconomy(FoundationalSector):
    pass


@dataclass
class ProvisioningRealm:
    pass


@dataclass
class StateRealm(ProvisioningRealm):
    government_effectiveness_index: float
    government_spending_in_gdp: float
    one_party_system: bool
    military_rule: bool
    leader_ideology: bool
    corruption_perception_index: float
    protests_and_riots_per_cap: float


@dataclass
class MarketRealm(ProvisioningRealm):
    market_concentration: float
    corporate_tax_rate: float
    lobbying_expenditure_in_gdp: float
    trade_penetration: float
    commercial_employment: float


@dataclass
class HouseholdRealm(ProvisioningRealm):
    household_consumption_in_gdp: float


@dataclass
class CommonsRealm(ProvisioningRealm):
    pass
