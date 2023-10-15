from dataclasses import dataclass


# internal:
# levels of need satisfaction by variable

@dataclass
class NeedSatisfaction:
    food: float
    water: float
    shelter: float
