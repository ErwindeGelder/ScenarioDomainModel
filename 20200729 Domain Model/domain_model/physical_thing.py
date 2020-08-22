""" Class PhysicalThing

Creation date: 2020 08 19
Author(s): Erwin de Gelder

Modifications:
2020 08 22: Add function to obtain properties from a dictionary.
"""

from abc import abstractmethod
from .quantitative_thing import QuantitativeThing, _quantitative_thing_props_from_json
from .type_checking import check_for_type


class PhysicalThing(QuantitativeThing):
    """ Class for modeling all physical things (both static and dynamic)

    A PhysicalThing is used to quantitatively describe a physical thing. This is
    an abstract class, so it is not possible to instantiate objects from this
    class.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        properties(dict): All properties of the physical thing.
    """
    @abstractmethod
    def __init__(self, properties: dict = None, **kwargs):
        if properties is None:
            properties = dict()
        check_for_type("properties", properties, dict)

        QuantitativeThing.__init__(self, **kwargs)
        self.properties = dict() if properties is None else properties

    def to_json(self) -> dict:
        thing = QuantitativeThing.to_json(self)
        thing["properties"] = self.properties
        return thing


def _physical_thing_props_from_json(json: dict) -> dict:
    props = dict(properties=json["properties"])
    props.update(_quantitative_thing_props_from_json(json))
    return props
