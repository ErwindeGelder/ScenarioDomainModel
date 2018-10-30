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
from typing import List


class StaticEnvironmentCategory(Default):
    """ Static environment category

    The static environment refers to the part of a scenario that does not change during a scenario. This includes
    geo-spatially stationary elements, such as the infrastructure layout, the road layout and the type of road. Also
    the presence of buildings near the road side that act as a view-blocking obstruction are considered part of the
    static environment.
    The StaticEnvironmentCategory only describes the static environment in qualitative terms.

    Attributes:
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the static environment category.
        description (str): A description of the static environment category. The objective of the description is to make
            the static environment category human interpretable.
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, uid, name, description, tags=None):
        # Check the types of the inputs
        if not isinstance(description, str):
            raise TypeError("Input 'description' should be of type <str> but is of type {0}.".format(type(description)))

        Default.__init__(self, uid, name, tags=tags)
        self.description = description

    def get_tags(self):
        return self.tags

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        attributes of StaticEnvironmentCategory to JSON.

        :return: dictionary that can be converted to a json file
        """
        static_environment_category = Default.to_json(self)
        static_environment_category["description"] = self.description
        return static_environment_category
