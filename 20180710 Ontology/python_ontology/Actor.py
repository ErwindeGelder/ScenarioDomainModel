from QualitativeActor import QualitativeActor


class Actor:
    def __init__(self, name, qualitative_actor, tags=None):
        self.name = name
        self.qualitative_actor = qualitative_actor  # type: QualitativeActor
        self.tags = [] if tags is None else tags

    def get_tags(self):
        tags = self.tags
        tags += self.qualitative_actor.tags
        return tags
