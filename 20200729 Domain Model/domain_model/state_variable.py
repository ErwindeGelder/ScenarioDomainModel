""" Class StateVariable

Creation date: 2018 11 05
Author(s): Erwin de Gelder

Modifications:
2019 03 14: StateVariables {LONGITUDINAL,LATERAL}_ROAD_POSITION added.
2019 10 11: Update of terminology.
2019 11 04: Add the heading as a possible state variable.
2020 10 29: Add units to state variable descriptions.
"""

from enum import Enum


class StateVariable(Enum):
    """ Enumeration for state variables.
    """
    LONGITUDINAL_POSITION = "x [m]"
    LATERAL_POSITION = "y [m]"
    SPEED = "v [m/s]"
    HEADING = "psi [rad]"
    YAWRATE = "psidot [rad/s]"
    LONGITUDINAL_ROAD_POSITION = "[ROAD_ID, ROAD_DISTANCE]"
    LATERAL_ROAD_POSITION = "[LANE_ID, LANE_OFFSET]"
    LON_TARGET = "[vabs [m/s], dx [m]]"
    LAT_TARGET = 'dy [m]'

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
