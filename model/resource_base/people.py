from dataclasses import dataclass


# add education, health.. human development components
@dataclass
class DemographicsAndUrbanization:
    population: float
    population_growth_rate: float
    ethnic_diversity: float
    urbanization_rate: float
    urbanization_level: float
    regional_income_disparity: float
