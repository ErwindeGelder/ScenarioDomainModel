""" Class QualitativeThing

Creation date: 2020 08 15
Author(s): Erwin de Gelder

Modifications:
"""

from abc import abstractmethod
from .thing import Thing
from .type_checking import check_for_type


class QualitativeThing(Thing):
    """ Thing that is used for most qualitative classes.

    Next to the attributes of Thing, a QualitativeThing also has a description
    that can be used to qualitatively describe the thing. This is an abstract
    class, so it is not possible to instantiate objects from this class.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        description(str): A string that qualitatively describes this thing.
    """
    @abstractmethod
    def __init__(self, description: str = "", **kwargs):
        # Check the types of the inputs
        check_for_type("description", description, str)

        self.description = description  # type: str
        Thing.__init__(self, **kwargs)

    def to_json(self) -> dict:
        thing = Thing.to_json(self)
        thing["description"] = self.description
        return thing
