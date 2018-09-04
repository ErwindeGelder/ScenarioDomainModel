class Scenario:
    def __init__(self, name, tstart, tend, actors=None, activities=None, static_environment=None, tags=None):
        self.name = name
        self.tstart = tstart
        self.tend = tend
        self.actors = [] if actors is None else actors
        self.activities = [] if activities is None else activities
        self.static_environment = static_environment
        self.tags = [] if tags is None else tags

    def get_tags(self):
        tags = self.tags
        tags += []
        return tags
