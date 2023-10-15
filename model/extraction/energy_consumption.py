from dataclasses import dataclass


# internal:
# energy use
# energy sources
# energy efficiency


@dataclass
class EnergySource:
    share_of_generation: float
    share_of_employment: float
    share_of_investment: float


@dataclass
class Oil(EnergySource):
    pass


@dataclass
class Coal(EnergySource):
    pass


@dataclass
class NaturalGas(EnergySource):
    pass


@dataclass
class Renewables(EnergySource):
    pass


@dataclass
class EnergyConsumption:
    energy_use: float
    oil: Oil
    coal: Coal
    natural_gas: NaturalGas
    renewables: Renewables
