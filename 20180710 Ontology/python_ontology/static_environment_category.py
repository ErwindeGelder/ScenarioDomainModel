from tags import Tag
from typing import List


class StaticEnvironmentCategory:
    """ Static environment category

    The static environment refers to the part of a scenario that does not change during a scenario. This includes
    geo-spatially stationary elements, such as the infrastructure layout, the road layout and the type of road. Also
    the presence of buildings near the road side that act as a view-blocking obstruction are considered part of the
    static environment.
    The StaticEnvironmentCategory only describes the static environment in qualitative terms.

    Attributes:
        name (str): A name that serves as a short description of the static environment category.
        description (str): A description of the static environment category. The objective of the description is to make
            the static environment category human interpretable.
        tags (List[Tag]): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, description, tags=None):
        self.name = name
        self.description = description
        self.tags = [] if tags is None else tags  # Type: List[Tag]

    def get_tags(self):
        return self.tags
