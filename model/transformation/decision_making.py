from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


@dataclass
class PoliticsAndState:
    government_effectiveness_index: float
    government_spending_in_gdp: float
    one_party_system: bool
    military_rule: bool
    leader_ideology: bool
    corruption_perception_index: float
    protests_and_riots_per_cap: float


@dataclass
class MediaStructure:
    voice_and_accountability_index: float
    media_freedom_index: float
    tv_consumption_share: float


@dataclass
class CorporatePower:
    market_concentration: float
    corporate_tax_rate: float
    lobbying_expenditure_in_gdp: float


@dataclass
class EconomicInequality:
    gini_index: float
    atkinson_index: float
    top_1_percent_income_share: float
    top_1_percent_wealth_share: float


@dataclass
class GenderInequality:
    women_parliament_share: float
    women_minister_share: float
    women_management_share: float
    women_labour_income_share: float
    gender_pay_gap: float


@dataclass
class EthnicInequality:
    ethnic_inequality_index: float
    ethnic_pay_gap: float
    minority_parliament_share: float
    minority_minister_share: float
    minority_management_share: float
    minority_labour_income_share: float


@dataclass
class CrimeAndDespair:
    suicide_rate: float
    homicides_per_capita: float
    property_crime_per_capita: float
    incarceration_rate: float


@dataclass
class ParticipationInGlobalInstitutions:
    un_voting_index: float


@dataclass
class ParticipationInRegionalInstitutions:
    pass


@dataclass
class ParticipationInWars:
    current_armed_conflicts: int
    armed_conflicts_initiated_share: float
    military_expenditure_in_gdp: float


@dataclass
class DecisionMaking:
    politics_and_state: PoliticsAndState
    media_structure: MediaStructure
    corporate_power: CorporatePower
    economic_inequality: EconomicInequality
    gender_inequality: GenderInequality
    ethnic_inequality: EthnicInequality
    crime_and_despair: CrimeAndDespair
    participation_in_global_institutions: ParticipationInGlobalInstitutions
    participation_in_regional_institutions: ParticipationInRegionalInstitutions
    participation_in_wars: ParticipationInWars
