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
from .dynamic_physical_thing import DynamicPhysicalThing, dynamic_physical_thing_from_json
from .dynamic_physical_thing_category import DynamicPhysicalThingCategory, \
    dynamic_physical_thing_category_from_json
from .event import Event, event_from_json
from .model import Constant, Linear, Spline3Knots, Sinusoidal, Splines, model_from_json
from .scenario import Scenario, scenario_from_json
from .scenario_category import ScenarioCategory, scenario_category_from_json
from .state import State, state_from_json
from .state_variable import StateVariable, state_variable_from_json
from .static_physical_thing import StaticPhysicalThing, static_physical_thing_from_json
from .static_physical_thing_category import StaticPhysicalThingCategory, \
    static_physical_thing_category_from_json
from .tags import Tag, tag_from_json
from .thing import DMObjects, get_empty_dm_object
