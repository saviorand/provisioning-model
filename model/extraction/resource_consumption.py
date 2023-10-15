from dataclasses import dataclass


@dataclass
class ResourceConsumption:
    wood: float
    water: float
    metal: float
    rare_earth: float
