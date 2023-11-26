from dataclasses import dataclass


@dataclass
class Country:
    alpha2: str
    alpha3: str
    name: str


@dataclass
class Stocks:
    wealth_per_capita: float
