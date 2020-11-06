"""
domain_model

The domain model is used to describe scenarios and scenario classes and the corresponding
attributes.
"""

# imports to make easy access possible after importing domain_model
from .activity import Activity, activity_from_json
from .activity_category import ActivityCategory, activity_category_from_json
from .actor import Actor, EgoVehicle, actor_from_json
from .actor_category import ActorCategory, VehicleType, actor_category_from_json
from .document_management import DocumentManagement
from .event import Event, event_from_json
from .model import Constant, Linear, Spline3Knots, Sinusoidal, Splines, model_from_json
from .physical_element import PhysicalElement, physical_element_from_json
from .physical_element_category import PhysicalElementCategory, physical_element_category_from_json
from .scenario import Scenario, scenario_from_json
from .scenario_category import ScenarioCategory, scenario_category_from_json
from .scenario_element import DMObjects, get_empty_dm_object
from .state import State, state_from_json
from .state_variable import StateVariable, state_variable_from_json
from .tags import Tag, tag_from_json
