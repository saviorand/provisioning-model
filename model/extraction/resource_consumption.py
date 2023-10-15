from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


@dataclass
class ResourceConsumption:
    wood: float
    water: float
    metal: float
    rare_earth: float
