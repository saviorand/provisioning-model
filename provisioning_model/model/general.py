from dataclasses import dataclass


@dataclass
class Country:
    id: int
    alpha2: str
    alpha3: str
    name: str
