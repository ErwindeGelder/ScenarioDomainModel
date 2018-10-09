class QualitativeActor:
    """ Qualitative description of an actor

    An actor is an element of a scenario acting on its own behalf. The ego vehicle and other road users are examples of
    actors in a scenario. The qualitative actor only describes the actor in qualitative terms.

    Attributes:
        name (str): A name that serves as a short description of the qualitative actor.
        type (str): The type of the actor. This should be of the list [Vehicle, Car, Passenger car (M1), Van (N1),
            Minivan, Truck, Lorry (N), Trailer (O), Bus (M2 M3), PTW (L), Motorcycle (L3), Moped (L1), VRU, Pedestrian,
            Cyclist, Personal Mobility Device]
        tags (List of str): The tags are used to determine whether a scenario falls into a scenarioClass.
    """
    def __init__(self, name, vehicle_type, tags=None):
        self.name = name
        self.type = vehicle_type
        self.tags = [] if tags is None else tags
