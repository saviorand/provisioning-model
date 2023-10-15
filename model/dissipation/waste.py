from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


# internal:
# volume of waste
# structure of waste
# recycling rate

# capital depreciation?

# external:
# import/export of waste

@dataclass
class Waste:
    volume: float
    structure: float
    recycling_rate: float
