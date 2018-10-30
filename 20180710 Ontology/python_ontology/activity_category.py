"""
Class ActivityCategory


Author
------
Erwin de Gelder

Creation
--------
30 Oct 2018

To do
-----

Modifications
-------------

"""


from default_class import Default
from model_ import Model
from enum import Enum
from typing import List
from tags_ import Tag


class ActivityCategory(Default):
    """ Category of activity

    An activity specified the evolution of a state over time. The activity category describes the activity in
    qualitative terms.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the activity category.
        model (Model): Parameter Model describes the relation between the states variables and the parameters that
            specify an activity.
        state (StateVariable): The state is the variable that describes the behavior of the activity. Moreover, the
            state is the output of the mode.
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, uid, name, model, state, tags=None):
        # Check the types of the inputs
        if not isinstance(model, Model):
            raise TypeError("Input 'model' should be of type <Model> but is of type {0}.".format(type(model)))
        if not isinstance(state, StateVariable):
            raise TypeError("Input 'state' should be of type <StateVariable> but is of type {0}.".format(type(state)))

        Default.__init__(self, uid, name, tags=tags)
        self.model = model  # type: Model
        self.state = state  # type: StateVariable

    def get_tags(self):
        return self.tags

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        attributes of ActivityCategory to JSON.

        :return: dictionary that can be converted to a json file
        """
        activity_category = Default.to_json(self)
        activity_category["model"] = self.model.to_json()
        activity_category["state"] = self.state.to_json()
        return activity_category


class StateVariable(Enum):
    """ Enumeration for state variables.
    """
    LONGITUDINAL_POSITION = "x"
    LATERAL_POSITION = "y"

    def to_json(self):
        state_variable = {"name": self.name, "value": self.value}
        return state_variable
