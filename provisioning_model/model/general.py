from dataclasses import dataclass


@dataclass
class Country:
    id: str
    alpha2: str
    alpha3: str
    name: str
