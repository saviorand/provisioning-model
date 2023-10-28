from dataclasses import dataclass


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
class ForeignRelations:
    participation_in_global_institutions: ParticipationInGlobalInstitutions
    participation_in_regional_institutions: ParticipationInRegionalInstitutions
    participation_in_wars: ParticipationInWars
