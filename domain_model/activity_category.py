""" Class ActivityCategory

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 05 11: Make code PEP8 compliant.
2019 11 19: Enable instantiation using JSON code.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 11: Update of terminology.
"""

import numpy as np
from .model import Model, model_from_json
from .thing import Thing
from .state_variable import StateVariable, state_variable_from_json
from .tags import tag_from_json
from .type_checking import check_for_type


class ActivityCategory(Thing):
    """ Category of activity

    An activity specified the evolution of a state over time. The activity
    category describes the activity in qualitative terms.

    Attributes:
        model (Model): Parameter Model describes the relation between the states
            variables and the parameters that specify an activity.
        state (StateVariable): The state is the variable that describes the
            behavior of the activity. Moreover, the state is the output of the
            mode.
        name (str): A name that serves as a short description of the activity
            category.
        uid (int): A unique ID.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
    """
    def __init__(self, model: Model, state: StateVariable, **kwargs):
        # Check the types of the inputs
        check_for_type("model", model, Model)
        check_for_type("state", state, StateVariable)

        Thing.__init__(self, **kwargs)
        self.model = model  # type: Model
        self.state = state  # type: StateVariable

    def fit(self, time: np.ndarray, data: np.ndarray, options: dict = None) -> dict:
        """ Fit the data to the model and return the parameters.

        The data is to be fit to the model that is set for this ActivityCategory
        and the resulting parameters are returned in a dictionary. See the fit
        method from Model for more details.

        :param time: the time instants of the data.
        :param data: the data that will be fit to the model.
        :param options: specify some model-specific options.
        :return: dictionary of the parameters.
        """
        return self.model.fit(time, data, options=options)

    def to_json(self) -> dict:
        """ Get JSON code of object.

        For storing scenarios into the database, the scenarios need to be
        converted to JSON. This method converts the attributes of
        ActivityCategory to JSON.

        :return: dictionary that can be converted to a json file.
        """
        activity_category = Thing.to_json(self)
        activity_category["model"] = self.model.to_json()
        activity_category["state"] = self.state.to_json()
        return activity_category


def activity_category_from_json(json: dict) -> ActivityCategory:
    """ Get ActivityCategory object from JSON code.

    It is assumed that the JSON code of the ActivityCategory is created using
    ActivityCategory.to_json().

    :param json: JSON code of ActorCategory.
    :return: ActivityCategory object.
    """
    model = model_from_json(json["model"])
    state = state_variable_from_json(json["state"])
    activity_category = ActivityCategory(model, state, name=json["name"], uid=int(json["id"]),
                                         tags=[tag_from_json(tag) for tag in json["tag"]])
    return activity_category
