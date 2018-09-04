class QualitativeActor:
    def __init__(self, name, vehicle_type, tags=None):
        self.name = name
        self.type = vehicle_type
        self.tags = [] if tags is None else tags
