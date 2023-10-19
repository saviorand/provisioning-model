from dataclasses import dataclass


# internal:
# how much is consumed of what (quantity), e.g. energy use
# composition, e.g. energy sources
# dynamics of consumption, what is growing and what declines

@dataclass
class EnergyUse:
    pass


@dataclass
class Employment:
    informal_employment: float
    unemployment: float
    average_working_hours: float
    wage_worker_share: float
    self_employed_share: float
    unpaid_family_worker_share: float
