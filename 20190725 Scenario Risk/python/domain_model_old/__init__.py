"""
domain_model

The domain model is used to describe scenarios and scenario classes and the corresponding
attributes.
"""

# imports to make easy access possible after importing domain_model
from .activity_category import ActivityCategory, activity_category_from_json
from .actor_category import ActorCategory, VehicleType, actor_category_from_json
from .activity import Activity, DetectedActivity, TriggeredActivity, activity_from_json
from .actor import Actor, EgoVehicle, actor_from_json
from .document_management import DocumentManagement
from .event import Event
from .model import Spline3Knots, Linear, Sinusoidal, Constant, BSplines, MultiBSplines, \
    model_from_json
from .scenario import Scenario, scenario_from_json
from .scenario_category import ScenarioCategory, scenario_category_from_json
from .state import State, state_from_json
from .state_variable import StateVariable, state_variable_from_json
from .static_environment_category import StaticEnvironmentCategory, Region, \
    stat_env_category_from_json
from .static_environment import StaticEnvironment, stat_env_from_json
from .tags import Tag, tag_from_json
