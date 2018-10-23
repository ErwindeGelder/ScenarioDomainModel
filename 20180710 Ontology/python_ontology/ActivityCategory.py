class ActivityCategory:
    """ Category of activity

    An activity specified the evolution of a state over time. The activity category describes the activity in
    qualitative terms.

    Attributes:
        name (str):
        state (str):
    """
    def __init__(self, name, model, state, tags=None):
        self.name = name
        self.model = model
        self.state = state
        self.tags = [] if tags is None else tags
