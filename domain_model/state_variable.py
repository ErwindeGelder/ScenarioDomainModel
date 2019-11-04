"""
Class StateVariable


Author
------
Erwin de Gelder

Creation
--------
05 Nov 2018

To do
-----

Modifications
-------------
14 Mar 2019: StateVariables {LONGITUDINAL,LATERAL}_ROAD_POSITION added.
11 Oct 2019: Update of terminology.
04 Nov 2019: Add the heading as a possible state variable.

"""

from enum import Enum


class StateVariable(Enum):
    """ Enumeration for state variables.
    """
    LONGITUDINAL_POSITION = "x"
    LATERAL_POSITION = "y"
    SPEED = "v"
    HEADING = "psi"
    LONGITUDINAL_ROAD_POSITION = "[ROAD_ID, ROAD_DISTANCE]"
    LATERAL_ROAD_POSITION = "[LANE_ID, LANE_OFFSET]"

    def to_json(self):
        """ When tag is exporting to JSON, this function is being called
        """
        state_variable = {"name": self.name, "value": self.value}
        return state_variable


def state_variable_from_json(json: dict) -> StateVariable:
    """ Get StateVariable object from JSON code

    It is assumed that the JSON code of the StateVariable is created using StateVariable.to_json().

    :param json: JSON code of StateVariable.
    :return: Tag object
    """

    return getattr(StateVariable, json["name"])
