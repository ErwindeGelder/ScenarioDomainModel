"""
Class Actor


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
from actor_category import ActorCategory, VehicleType
from tags_ import Tag
import json
from typing import List


class Actor(Default):
    """ Category of actor

    An actor is an agent in a scenario acting on its own behalf. "Ego vehicle" and "Other Road User" are types of
    actors in a scenario. The actor category only describes the actor in qualitative terms.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the qualitative actor.
        actor_category (ActorCategory): Specifying the category to which the actor belongs to.
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, uid, name, actor_category, tags=None):
        # Check the types of the inputs
        if not isinstance(actor_category, ActorCategory):
            raise TypeError("Input 'vehicle_type' should be of type <ActorCategory> but is of type {0}.".
                            format(type(actor_category)))

        Default.__init__(self, uid, name, tags=tags)
        self.actor_category = actor_category  # type: ActorCategory

    def get_tags(self):
        tags = self.tags
        tags += self.actor_category.get_tags()
        return tags

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        attributes of Actor to JSON.

        :return: dictionary that can be converted to a json file
        """
        actor = Default.to_json(self)
        actor["actor_category"] = self.actor_category.name
        return actor


class EgoVehicle(Actor):
    """ Special actor: Ego vehicle

    The ego vehicle is similarly defined as an actor. The only difference is that it contains the tag "Ego vehicle".

    """
    def __init__(self, uid, name, actor_category, tags=None):
        Actor.__init__(self, uid, name, actor_category, tags=tags)
        self.tags += [Tag.EGO_VEHICLE]


# An example to illustrate how an actor can be instantiated.
if __name__ == '__main__':
    # Create an actor category that describes the actor in qualitative terms.
    ac = ActorCategory(0, "sedan", VehicleType.PASSENGER_CAR_M1, tags=[Tag.ACTOR_TYPE_PASSENGER_CAR_M1])

    # Create an actor that is the ego vehicle.
    ego = EgoVehicle(0, "Ego", ac)

    # Show the tags that are associated with the actor.
    print("Tags of the actor:")
    for t in ego.get_tags():
        print(" - {:s}".format(t.name))

    # Show the JSON code when the ActorCategory is exported to JSON
    print()
    print("JSON code for the ActorCategory:")
    print(ac.to_json())
    print(json.dumps(ac.to_json(), indent=4))

    # Show the JSON code when this actor is exported to JSON
    print()
    print("JSON code for the Actor:")
    print(json.dumps(ego.to_json(), indent=4))
