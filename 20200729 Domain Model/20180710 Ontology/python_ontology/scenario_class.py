"""
Class Activity


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
from tags import Tag
from typing import List, Tuple
from static_environment_category import StaticEnvironmentCategory
from actor_category import ActorCategory, VehicleType
from activity_category import ActivityCategory, StateVariable
from model import Model
import json
import os


class ScenarioClass(Default):
    """ ScenarioClass - A qualitative description

    Although a scenario is a quantitative description, there also exists a qualitative description of a scenario. We
    refer to the qualitative description of a scenario as a scenario class. The qualitative description can be regarded
    as an abstraction of the quantitative scenario.

    Scenarios fall into scenario classes. Multiple scenarios can fall into a single scenario class. On the other hand,
    a scenario may fall into one or multiple scenario classes. As an example, consider all scenarios that occur during
    the day. These scenarios fall into the scenario class "Day". Similarly, all scenarios with rain fall into the
    scenario class "Rain", see Figure 2. A scenario that occurs during the night without rain does not fall into any of
    the previously defined scenario classes. Likewise, a scenario that occurs during the day with rain falls into both
    scenario classes "Day" and "Rain".

    A scenario class can fall into another scenario class. For example, when we continue our previous example and
    consider the scenario class "Day and rain", this scenario class falls into the scenario classes "Day" and "Rain.
    Also, a scenario that occurs during the day with rain now falls into three scenario classes: "Day", "Rain", and
    "Day and rain".

    Attributes:
        name (str): A name that serves as a short description of the scenario class.
        description (str): A description of the scenario class. The objective of the description is to make the scenario
            class human interpretable.
        image (str): Path to image that schematically shows the class.
        static_environment (StaticEnvironmentCategory): Static environment of the Scenario.
        activities (List[ActivityCategory]): List of activities that are used for this Scenario.
        actors (List[ActorCategory]): List of actors that participate in the Scenario.
        acts (List[Tuple[ActorCategory, ActivityCategory]]): The acts describe which actors perform which activities.
            The actors and activities that are used in acts should also be passed with the actors and activities
            arguments. If not, a warning will be shown and the corresponding actor/activity will be added to the list of
            actors/activities.
        uid (int): A unique ID.
        tags (List[Tag]): A list of tags that formally defines the scenario class. These tags determine whether
            scenarios fall into this scenario class or not.
    """
    def __init__(self, name, description, image, static_environment, activities=None, actors=None, acts=None,
                 uid=-1, tags=None, verbose=True):
        # Check the types of the inputs
        if not isinstance(description, str):
            raise TypeError("Input 'description' should be of type <str> but is of type {0}.".format(type(description)))
        if not isinstance(image, str):
            raise TypeError("Input 'image' should be of type <str> but is of type {0}.".format(type(image)))
        if not isinstance(static_environment, StaticEnvironmentCategory):
            raise TypeError("Input 'static_environment' should be of type <StaticEnvironmentCategory> but is of type " +
                            "{0}.".format(type(static_environment)))
        if activities is not None:
            if not isinstance(activities, List):
                raise TypeError("Input 'activities' should be of type <List> but is of type {0}.".
                                format(type(activities)))
            for activity in activities:
                if not isinstance(activity, ActivityCategory):
                    raise TypeError("Items of input 'activities' should be of type <ActivityCategory> but at least " +
                                    "one item is of type {0}.".format(type(activity)))
        if actors is not None:
            if not isinstance(actors, List):
                raise TypeError("Input 'actors' should be of type <List> but is of type {0}.".format(type(actors)))
            for actor in actors:
                if not isinstance(actor, ActorCategory):
                    raise TypeError("Items of input 'actors' should be of type <ActorCategory> but at least one item " +
                                    "is of type {0}.".format(type(actor)))
        if acts is not None:
            if not isinstance(acts, List):
                raise TypeError("Input 'acts' should be of type <List> but is of type {0}.".format(type(acts)))
            for act in acts:
                if not isinstance(act, tuple):
                    raise TypeError("Items of input 'acts' should be of type <tuple> but at least one item is of type" +
                                    " {0}.".format(type(act)))
                if not len(act) == 2:
                    raise TypeError("Items of input 'acts' should be tuples with length 2 but at least one item has " +
                                    "length {:d}.".format(len(act)))
                if not isinstance(act[0], ActorCategory):
                    raise TypeError("First item of a tuple from the list of 'acts' should be of type <ActorCategory> " +
                                    "but this first item is at least for one tuple of type {0}".format(type(act[0])))
                if not isinstance(act[1], ActivityCategory):
                    raise TypeError("Second item of a tuple from the list of 'acts' should be of type " +
                                    "<ActivityCategory> but this second item is at least for one tuple of type {0}".
                                    format(type(act[1])))

        Default.__init__(self, uid, name, tags=tags)
        self.description = description
        self.image = image
        self.static_environment = static_environment  # Type: StaticEnvironmentCategory
        self.activities = [] if activities is None else activities  # Type: List[ActivityCategory]
        self.actors = [] if actors is None else actors  # Type: List[ActorCategory]
        self.acts = [] if acts is None else acts  # Type: List[Tuple[ActorCategory, ActivityCategory]]

        # Check whether the actors/activities defined with the acts are already listed. If not, the corresponding
        # actor/activity will be added and a warning will be shown.
        for actor, activity in self.acts:
            if actor not in self.actors:
                if verbose:
                    print("Actor with name '{:s}' is used with acts but not defined in the list of actors."
                          .format(actor.name))
                    print("Therefore, the actor is added to the list of actors.")
                self.actors.append(actor)
            if activity not in self.activities:
                if verbose:
                    print("Activity with name '{:s}' is used with acts but not defined in the list of activities."
                          .format(activity.name))
                    print("Therefore, the activity is added to the list of activities.")
                self.activities.append(activity)

        # Some parameters
        self.maxprintlength = 80  # Maximum number of characters that are used when printing the general description

    def __str__(self):
        """ Method that will be called when printing the scenario class

        :return: string to print
        """

        # Show the name
        string = "Name: {:s}\n".format(self.name)

        # Show the description of the scenario class
        string += "Description:\n"
        words = self.description.split(' ')
        line = ""
        for word in words:
            if len(line) + len(word) < 80:
                line += " {:s}".format(word)
            else:
                string += "{:s}\n".format(line)
                line = " {:s}".format(word)
        if len(line) > 0:
            string += "{:s}\n".format(line)

        # Show the tags
        string += "Tags:\n"
        if self.tags is None:
            string += "Not available\n"
        else:
            for tag in self.tags:
                string += {" - {:s}".format(tag.name)}
        return string

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        attributes of Scenario to JSON.

        :return: dictionary that can be converted to a json file
        """
        scenario_class = Default.to_json(self)
        scenario_class["description"] = self.description
        scenario_class["image"] = self.image
        scenario_class["static_environment_category"] = self.static_environment.to_json()
        scenario_class["actor_category"] = [actor.to_json() for actor in self.actors]
        scenario_class["activity_category"] = [activity.to_json() for activity in self.activities]
        scenario_class["acts"] = [{"actor": actor.name, "activity": activity.name} for actor, activity in self.acts]
        return scenario_class


# Add an example
if __name__ == '__main__':
    sc_name = "Ego vehicle approaching slower lead vehicle"
    sc_desc = "Another vehicle is driving in front of the ego vehicle at a slower speed. As a result, the other " + \
              "vehicle appears in the ego vehicle's field of view. The ego vehicle might brake to avoid a " + \
              "collision. Another possibility for the ego vehicle is to perform a lane change if this is " + \
              "possible. The reason for the other vehicle to drive slower is e.g., due to a traffic jam ahead."
    sc_image = ""

    # Define the static environment
    sc_static_environment = StaticEnvironmentCategory("Anything",
                                                      "No further details are specified for the static environment.")

    # Define the actors
    ego_vehicle = ActorCategory("Ego", VehicleType.VEHICLE, tags=[Tag.EGO_VEHICLE])
    target_vehicle = ActorCategory("Target", VehicleType.VEHICLE, tags=[Tag.INIT_STATE_LONG_POS_IN_FRONT_OF_EGO,
                                                                        Tag.INIT_STATE_DIRECTION_SAME_AS_EGO,
                                                                        Tag.INIT_STATE_LAT_POS_SAME_LANE,
                                                                        Tag.ACTOR_TYPE_VEHICLE])

    # Define the activities
    going_straight = ActivityCategory("Going straight", Model("NA"), StateVariable.LATERAL_POSITION,
                                      tags=[Tag.VEH_LAT_ACT_LANE_FOLLOWING])
    going_forward = ActivityCategory("Going forward", Model("NA"), StateVariable.LONGITUDINAL_POSITION,
                                     tags=[Tag.VEH_LONG_ACT_DRIVING_FORWARD])

    sc = ScenarioClass(sc_name, sc_desc, sc_image, sc_static_environment,
                       activities=[going_straight], actors=[ego_vehicle, target_vehicle],
                       acts=[(target_vehicle, going_straight), (target_vehicle, going_forward)])

    # Show the JSON code when this scenario class is exported to JSON
    print()
    print("JSON code for the Scenario:")
    print(json.dumps(sc.to_json(), indent=4))
    with open(os.path.join('examples', 'scenario_class.json'), "w") as f:
        json.dump(sc.to_json(), f, indent=4)
