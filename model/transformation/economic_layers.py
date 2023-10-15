from dataclasses import dataclass


@dataclass
class EconomicLayer:
    value_added_share: float
    employment_share: float
    compensation_share: float
    investment_share: float


@dataclass
class Core(EconomicLayer):
    pass


@dataclass
class FoundationalMaterial(EconomicLayer):
    pass


@dataclass
class FoundationalProvidential(EconomicLayer):
    pass


@dataclass
class Overlooked(EconomicLayer):
    pass


@dataclass
class Competitive(EconomicLayer):
    pass


@dataclass
class Foundational:
    material: FoundationalMaterial
    providential: FoundationalProvidential


@dataclass
class Layers:
    core: Core
    foundational: Foundational
    overlooked: Overlooked
    competitive: Competitive
