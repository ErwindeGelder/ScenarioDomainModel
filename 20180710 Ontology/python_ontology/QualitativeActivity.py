class QualitativeActivity:
    def __init__(self, name, model, state, tags=None):
        self.name = name
        self.model = model
        self.state = state
        self.tags = [] if tags is None else tags
