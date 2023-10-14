import reflex as rx


# demographics, ethnic diversity
# natural endowment: land, water, climate, minerals, energy
# artificial endowment: urbanization, where is capital located, infrastructure, education

class Urbanization(rx.Model, table=True):
    urbanization_rate: float
    urbanization_level: float


class NaturalResources(rx.Model, table=True):
    land_area: float
    rare_minerals_per_cap: float
    clean_water_per_cap: float
