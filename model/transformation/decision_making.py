import reflex as rx


class PoliticsAndState(rx.Model, table=True):
    government_effectiveness_index: float
    government_spending_in_gdp: float
    one_party_system: bool
    military_rule: bool
    leader_ideology: bool
    corruption_perception_index: float
    protests_and_riots_per_cap: float


class MediaStructure(rx.Model, table=True):
    voice_and_accountability_index: float
    media_freedom_index: float
    tv_consumption_share: float


class CorporatePower(rx.Model, table=True):
    market_concentration: float
    corporate_tax_rate: float
    lobbying_expenditure_in_gdp: float


class EconomicInequality(rx.Model, table=True):
    gini_index: float
    atkinson_index: float
    top_1_percent_income_share: float
    top_1_percent_wealth_share: float


class GenderInequality(rx.Model, table=True):
    women_parliament_share: float
    women_minister_share: float
    women_management_share: float
    women_labour_income_share: float
    gender_pay_gap: float


class EthnicInequality(rx.Model, table=True):
    ethnic_inequality_index: float
    ethnic_pay_gap: float
    minority_parliament_share: float
    minority_minister_share: float
    minority_management_share: float
    minority_labour_income_share: float


class CrimeAndDespair(rx.Model, table=True):
    suicide_rate: float
    homicides_per_capita: float
    property_crime_per_capita: float
    incarceration_rate: float


class ParticipationInGlobalInstitutions(rx.Model, table=True):
    un_voting_index: float


class ParticipationInRegionalInstitutions(rx.Model, table=True):
    pass


class ParticipationInWars(rx.Model, table=True):
    current_armed_conflicts: int
    armed_conflicts_initiated_share: float
    military_expenditure_in_gdp: float


class DecisionMaking(rx.Model):
    politics_and_state: PoliticsAndState
    media_structure: MediaStructure
    corporate_power: CorporatePower
    economic_inequality: EconomicInequality
    gender_inequality: GenderInequality
    racial_inequality: RacialInequality
    crime_and_despair: CrimeAndDespair
    participation_in_global_institutions: ParticipationInGlobalInstitutions
    participation_in_regional_institutions: ParticipationInRegionalInstitutions
    participation_in_wars: ParticipationInWars
