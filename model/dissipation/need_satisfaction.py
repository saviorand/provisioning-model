from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


# internal:
# levels of need satisfaction by variable

@dataclass
class NeedSatisfaction:
    food: float
    water: float
    shelter: float
