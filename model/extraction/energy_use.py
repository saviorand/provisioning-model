from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


# internal:
# energy use
# energy sources
# energy efficiency

@dataclass
class EnergyConsumption:
    energy_use: float
