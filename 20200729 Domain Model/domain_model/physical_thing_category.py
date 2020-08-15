""" Class PhysicalThingCategory

Creation date: 2020 08 15
Author(s): Erwin de Gelder

Modifications:
"""

from abc import abstractmethod
from .qualitative_thing import QualitativeThing


class PhysicalThingCategory(QualitativeThing):
    """ PhysicalThingCategory: Category of a physical thing

    A PhysicalThingCategory is used to qualitatively describe a physical thing.
    This is an abstract class, so it is not possible to instantiate objects from
    this class.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        description(str): A string that qualitatively describes this thing.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        QualitativeThing.__init__(self, **kwargs)
