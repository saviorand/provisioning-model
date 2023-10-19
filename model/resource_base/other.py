from dataclasses import dataclass


@dataclass
class NaturalResourceBase:
    land_area: float
    forest_area: float
    fossil_fuel_reserves: float
    rare_minerals_per_cap: float
    clean_water_per_cap: float
    climate_index: float
