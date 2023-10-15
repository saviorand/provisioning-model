from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


# demographics, ethnic diversity
# natural endowment: land, water, climate, minerals, energy
# artificial endowment: urbanization, where is capital located, infrastructure, education
@dataclass
class Urbanization():
    urbanization_rate: float
    urbanization_level: float


@dataclass
class NaturalResources():
    land_area: float
    rare_minerals_per_cap: float
    clean_water_per_cap: float


@dataclass
class InitialConditions():
    natural_resources: NaturalResources
    urbanization: Urbanization
