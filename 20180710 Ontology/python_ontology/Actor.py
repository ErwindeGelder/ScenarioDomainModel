from ActorCategory import ActorCategory
import json


class Actor:
    """ Category of actor

    An actor is an agent in a scenario acting on its own behalf. "Ego vehicle" and "Other Road User" are types of
    actors in a scenario. The actor category only describes the actor in qualitative terms.

    Attributes:
        name (str): A name that serves as a short description of the qualitative actor.
        actor_category (ActorCategory): Specifying the category to which the actor belongs to.
        tags (List of str): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, actor_category, tags=None):
        self.name = name
        self.actor_category = actor_category  # type: ActorCategory
        self.tags = [] if tags is None else tags

    def get_tags(self):
        tags = self.tags
        tags += self.actor_category.get_tags()
        return tags

    def to_json(self):
        """

        :return: dictionary that can be converted to a json file
        """
        actor = {"name": self.name,
                 "actor_category": self.actor_category.name,
                 "tag": self.tags}
        return actor


class EgoVehicle(Actor):
    """ Special actor: Ego vehicle

    The ego vehicle is similarly defined as an actor. The only difference is that it contains the tag "Ego vehicle".

    """
    def __init__(self, name, actor_category, tags=None):
        Actor.__init__(self, name, actor_category, tags=tags)
        self.tags += ['Ego vehicle']


# An example to illustrate how an actor can be instantiated.
if __name__ == '__main__':
    # Create an actor category that describes the actor in qualitative terms.
    ac = ActorCategory("Sedan", "Passenger car (M1)", tags=["Passenger car (M1)"])

    # Create an actor that is the ego vehicle.
    ego = EgoVehicle("Ego", ac)

    # Show the tags that are associated with the actor.
    print("Tags of the actor:")
    for tag in ego.get_tags():
        print(" - {:s}".format(tag))

    # Show the JSON code when this actor is exported to json
    print()
    print("JSON code for the actor:")
    print(json.dumps(ego.to_json(), indent=4))
