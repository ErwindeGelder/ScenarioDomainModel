from tags import Tag
from typing import List, Tuple
from static_environment_category import StaticEnvironmentCategory
from actor_category import ActorCategory, VehicleType
from activity_category import ActivityCategory
from model import Model


class ScenarioClass:
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
        static_environment (StaticEnvironmentCategory): Static environment of the ScenarioClass.
        activities (List[ActivityCategory]): List of activities that are used for this ScenarioClass.
        actors (List[ActorCategory]): List of actors that participate in the ScenarioClass.
        acts (List[Tuple[ActorCategory, ActivityCategory]]): The acts describe which actors perform which activities.
        tags (List[Tag]): A list of tags that formally defines the scenario class. These tags determine whether
            scenarios fall into this scenario class or not.
    """
    def __init__(self, name, description, image, static_environment, activities=None, actors=None, acts=None,
                 tags=None):
        self.name = name
        self.description = description
        self.image = image
        self.static_environment = static_environment  # Type: StaticEnvironmentCategory
        self.activities = [] if activities is None else activities  # Type: List[ActivityCategory]
        self.actors = [] if actors is None else actors  # Type: List[ActorCategory]
        self.acts = [] if acts is None else acts  # Type: List[Tuple[ActorCategory, ActivityCategory]]
        self.tags = [] if tags is None else tags  # Type: List[Tag]

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
    target_vehicle = ActorCategory("Target", VehicleType.VEHICLE)

    # Define the activities
    going_straight = ActivityCategory("Going straight", Model("NA"), "x")

    sc = ScenarioClass(sc_name, sc_desc, sc_image, sc_static_environment)
    print(sc)
