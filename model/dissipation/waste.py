from dataclasses import dataclass


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
