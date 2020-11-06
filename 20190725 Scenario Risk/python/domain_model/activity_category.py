""" Class ActivityCategory

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 05 11: Make code PEP8 compliant.
2019 11 19: Enable instantiation using JSON code.
2019 05 22: Make use of type_checking.py to shorten the initialization.
2019 10 11: Update of terminology.
2020 08 16: Make ActivityCategory a subclass of QualitativeElement.
2020 08 25: Add function to obtain properties from a dictionary.
2020 10 04: Change way of creating object from JSON code.
2020 10 30: For using options for the fit functions, use **kwargs instead of options.
"""

import numpy as np
from .model import Model, _model_from_json
from .qualitative_element import QualitativeElement, _qualitative_element_props_from_json
from .scenario_element import DMObjects, _object_from_json, _attributes_from_json
from .state_variable import StateVariable, state_variable_from_json
from .type_checking import check_for_type


class ActivityCategory(QualitativeElement):
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
        description(str): A string that qualitatively describes the activity
            category.
    """
    def __init__(self, model: Model, state: StateVariable, **kwargs):
        # Check the types of the inputs
        check_for_type("model", model, Model)
        check_for_type("state", state, StateVariable)

        QualitativeElement.__init__(self, **kwargs)
        self.model = model  # type: Model
        self.state = state  # type: StateVariable

    def fit(self, time: np.ndarray, data: np.ndarray, **kwargs) -> dict:
        """ Fit the data to the model and return the parameters.

        The data is to be fit to the model that is set for this ActivityCategory
        and the resulting parameters are returned in a dictionary. See the fit
        method from Model for more details.

        :param time: the time instants of the data.
        :param data: the data that will be fit to the model.
        :param kwargs: specify some model-specific options.
        :return: dictionary of the parameters.
        """
        return self.model.fit(time, data, **kwargs)

    def to_json(self) -> dict:
        activity_category = QualitativeElement.to_json(self)
        activity_category["model"] = {"name": self.model.name, "uid": self.model.uid}
        activity_category["state"] = self.state.to_json()
        return activity_category

    def to_json_full(self) -> dict:
        activity_category = self.to_json()
        activity_category["model"] = self.model.to_json_full()
        return activity_category


def _activity_category_props_from_json(json: dict, attribute_objects: DMObjects,
                                       model: Model = None) -> dict:
    props = dict(state=state_variable_from_json(json["state"]))
    props.update(_qualitative_element_props_from_json(json))
    props.update(_attributes_from_json(json, attribute_objects,
                                       dict(model=(_model_from_json, "model")), model=model))
    return props


def _activity_category_from_json(json: dict, attribute_objects: DMObjects, model: Model = None) \
        -> ActivityCategory:
    return ActivityCategory(**_activity_category_props_from_json(json, attribute_objects, model))


def activity_category_from_json(json: dict, attribute_objects: DMObjects = None,
                                model: Model = None) -> ActivityCategory:
    """ Get ActivityCategory object from JSON code.

    It is assumed that the JSON code of the ActivityCategory is created using
    ActivityCategory.to_json(). Hence, the Model needs to be fully defined
    instead of only the unique ID. Alternatively, the Model can be passed as
    optional argument. In that case, the Model does not need to be defined in
    the JSON code.

    :param json: JSON code of ActorCategory.
    :param attribute_objects: A structure for storing all objects (optional).
    :param model: If given, it will not be based on the JSON code.
    :return: ActivityCategory object.
    """
    return _object_from_json(json, _activity_category_from_json, "activity_category",
                             attribute_objects, model=model)
