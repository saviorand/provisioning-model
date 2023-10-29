from dataclasses import dataclass


@dataclass
class Country:
    alpha2: str
    alpha3: str
    name: str
