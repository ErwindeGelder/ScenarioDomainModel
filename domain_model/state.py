""" Class State

Creation date: 2018 03 14
Author(s): Erwin de Gelder

Modifications:
2019 05 22: Make use of type_checking.py to shorten the initialization.
"""

from typing import Union, List
from .activity_category import StateVariable, state_variable_from_json
from .type_checking import check_for_type, check_for_list


class State:
    """ Describing the value of a StateVariable.

    This class is used to describe the initial state or the desired state of an
    actor.

    Attributes:
        state_variable(StateVariable): The state variable that is defined.
        values(List[float]): The actual value(s) of the state variable.
    """
    def __init__(self, state_variable: StateVariable, values: Union[List[float], float]):
        # Check the types of the inputs
        check_for_type("state_variable", state_variable, StateVariable)
        values = float(values) if isinstance(values, int) else values
        values = [values] if isinstance(values, float) else values
        check_for_list("values", values, (int, float), can_be_none=False, at_least_one=True)

        # Assign attributes
        self.state_variable = state_variable  # type: StateVariable
        self.value = values                   # type: List[float]

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of Event to JSON.

        :return: dictionary that can be converted to a json file.
        """
        return {"state_variable": self.state_variable.to_json(),
                "value": self.value}

    def to_json_full(self):
        """ Same as to_json

        It might be different from to_json for future releases, when the state
        is not fully described in to_json().
        """
        return self.to_json()


def state_from_json(json: dict) -> State:
    """ Get State object from JSON code

    It is assumed that all the attributes are fully defined.

    :param json: JSON code of Actor.
    :return: State object.
    """
    state_variable = state_variable_from_json(json["state_variable"])
    value = json["value"]
    return State(state_variable, value)
