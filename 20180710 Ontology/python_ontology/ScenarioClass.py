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
        tags (dict): A list of tags that formally defines the scenario class. These tags determine whether scenarios
            fall into this scenario class or not.
    """
    def __init__(self, name: str, description: str, tags=None):
        self.name = name
        self.description = description
        self.tags = {}
        if tags is not None:
            self.tags = tags

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
            for key, item in self.tags.items():
                length = len(key)
                string += "{:s}: ".format(key)
                if isinstance(item, dict):
                    for i, (key2, item2) in enumerate(item.items()):
                        length2 = len(key2)
                        if i > 0:
                            string += "{:{w}s}".format("", w=length+2)
                        string += "{:s}: ".format(key2)
                        if isinstance(item2, dict):
                            for j, (key3, item3) in enumerate(item2.items()):
                                if j > 0:
                                    string += "{:{w}s}".format("", w=length+length2+4)
                                string += "{:s}: {:s}\n".format(key3, item3)
                        else:
                            string += "{:s}\n".format(item2)
        return string


# Add an example
if __name__ == '__main__':
    scname = "Gap closing"
    scdesc = "Another vehicle is driving in front of the ego vehicle at a slower speed. As a result, the other " + \
             "vehicle appears in the ego vehicle's field of view. The ego vehicle might brake to avoid a " + \
             "collision. Another possibility for the ego vehicle is to perform a lane change if this is " + \
             "possible. The reason for the other vehicle to drive slower is e.g., due to a traffic jam ahead."
    sctags = {"Ego": {"Vehicle lateral activity": "Going straight",
                      "Vehicle longitudinal activity": "Driving forward"},
              "Actor": {"Road user type": "Vehicle",
                        "Initial state": {"Direction": "Same as ego",
                                          "Lateral position": "Same as ego",
                                          "Longitudinal position": "In front of ego"},
                        "Lead vehicle": {"Appearing": "Gap-closing"},
                        "Vehicle lateral activity": "Going straight",
                        "Vehicle longitudinal activity": "Driving forward"}}
    sc = ScenarioClass(name=scname, description=scdesc, tags=sctags)
    print(sc)
