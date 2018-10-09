class Scenario:
    """ Scenario - either a real-world scenario or a test case

    A scenario is a quantitative description of the ego vehicle, its activities and/or goals, its dynamic environment
    (consisting of traffic environment and conditions) and its static environment. From the perspective of the ego
    vehicle, a scenario contains all relevant events.

    Attributes:
        name (str): A name that serves as a short description of the scenario.
        tstart (float): The start time of the scenario. Part of the time intervals of the activities might be before the
            start time of the scenario.
        tend (float): The end time of the scenario. Part of the time interval of the activities might be after the end
            time of the scenario.
        actors (List of Actor): Actors that are participating in this scenario. This list should always include the ego
            vehicle.
        activities (List of Activity): Activities that are relevant for this scenario.
        actor_activity_links
    """

    def __init__(self, name, tstart, tend, actors=None, activities=None, actor_activity_links=None,
                 static_environment=None, tags=None):
        self.name = name
        self.tstart = tstart
        self.tend = tend
        self.actors = [] if actors is None else actors
        self.activities = [] if activities is None else activities
        self.actor_activity_links = [] if actor_activity_links is None else actor_activity_links
        self.static_environment = static_environment
        self.tags = [] if tags is None else tags

    def get_tags(self):
        tags = self.tags
        tags += []
        return tags
