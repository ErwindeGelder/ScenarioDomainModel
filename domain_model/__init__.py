"""
domain_model

The domain model is used to describe scenarios and scenario classes and the corresponding
attributes.
"""

# imports to make easy access possible after importing domain_model
from .activity_category import ActivityCategory
from .actor_category import ActorCategory, VehicleType
from .model import Spline3Knots, Linear, Sinusoidal, Constant
from .activity import Activity, DetectedActivity, TriggeredActivity
from .actor import Actor, EgoVehicle
from .scenario import Scenario
from .scenario_category import ScenarioCategory
from .tags import Tag
from .state_variable import StateVariable
from .static_environment_category import StaticEnvironmentCategory, Region
from .static_environment import StaticEnvironment
from .event import Event
from .state import State
