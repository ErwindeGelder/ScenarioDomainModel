"""
DefaultClass


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
from typing import List
from tags import Tag


class DefaultClass:
    """ DefaultClass that is used for most classes.

    Because most classes contain the attributes 'name' and 'tags', a default class is created that contains these
    attributes. This class also does type checking for these attributes. This class also supports conversion to JSON.

    Attributes:
        name (str): A name that serves as a short description of the actor category.
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, tags=None):
        # Check the types of the inputs
        if not isinstance(name, str):
            raise TypeError("Input 'name' should be of type <str> but is of type {0}.".format(type(name)))
        if tags is not None:
            if not isinstance(tags, List):
                raise TypeError("Input 'tags' should be of type <List> but is of type {0}.".format(type(tags)))
            for tag in tags:
                if not isinstance(tag, Tag):
                    raise TypeError("Items of input 'tags' should be of type <Tag> but at least one item is of type " +
                                    "{0}.".format(type(tag)))

        self.name = name  # type: str
        self.tags = [] if tags is None else tags  # type: List[Tag]

    def to_json(self):
        """ to_json

        For storing scenarios into the database, the scenarios need to be converted to JSON. This method converts the
        default attributes to JSON.

        :return: dictionary that can be converted to a json file
        """
        default_class = {"name": self.name,
                         "tag": [tag.to_json() for tag in self.tags]}
        return default_class
