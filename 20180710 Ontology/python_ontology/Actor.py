import matplotlib.pyplot as plt
from QualitativeActor import QualitativeActor
from Activity import Activity
from typing import List


class Actor:
    def __init__(self, name, qualitative_actor, list_of_activity, tags=None):
        self.name = name
        self.qualitative_actor = qualitative_actor  # type: QualitativeActor
        self.list_of_activity = list_of_activity  # type: List[Activity]
        self.tags = [] if tags is None else tags

    def plot_state(self, state, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        for activity in self.list_of_activity:
            if activity.qualitative_activity.state == state:
                activity.plot_state(ax=ax)

    def plot_state_dot(self, state, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        for activity in self.list_of_activity:
            if activity.qualitative_activity.state == state:
                activity.plot_state_dot(ax=ax)

    def get_tags(self):
        tags = self.tags
        tags += self.qualitative_actor.tags
        for activity in self.list_of_activity:
            tags += activity.qualitative_activity.tags
        return tags
