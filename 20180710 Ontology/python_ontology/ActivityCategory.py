from Model import Model


class ActivityCategory:
    """ Category of activity

    An activity specified the evolution of a state over time. The activity category describes the activity in
    qualitative terms.

    Attributes:
        name (str): A name that serves as a short description of the activity category.
        model (Model): Parameter Model describes the relation between the states variables and the parameters that
            specify an activity.
        state (str): The state is the variable that describes the behavior of the activity. Moreover, the state is the
            output of the mode.
        tags (List of str): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, model, state, tags=None):
        self.name = name
        self.model = model
        self.state = state
        self.tags = [] if tags is None else tags

    def get_tags(self):
        return self.tags
