from dataclasses import dataclass
from enum import Enum


# internal:
# energy use
# energy sources
# energy efficiency

@dataclass
class EnergyConsumption():
    energy_use: float

# class Alignment(str, Enum):
#     LAWFUL_GOOD = "lawful_good"
#     NEUTRAL_GOOD = "neutral_good"
#     CHAOTIC_GOOD = "chaotic_good"
#     LAWFUL_NEUTRAL = "lawful_neutral"
#     TRUE_NEUTRAL = "true_neutral"
#     CHAOTIC_NEUTRAL = "chaotic_neutral"
#     LAWFUL_EVIL = "lawful_evil"
#     NEUTRAL_EVIL = "neutral_evil"
#     CHAOTIC_EVIL = "chaotic_evil"


# @dataclass
# class Adventurer:
#     """A person often late for dinner but with a tale or two to tell.
#
#     Attributes:
#         name (str): Name of this adventurer
#         profession (str): Profession of this adventurer
#         level (int): Level of this adventurer
#         alignment (Alignment): Alignment of this adventurer
#     """
#
#     name: str
#     profession: str
#     level: int
#     alignment: Alignment
