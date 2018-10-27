import numpy as np


class Model:
    """ Model

    Parameter Model describes the relation between the states variables and the parameters that specify an activity.

    Example :
    x = a * t

    In this case,
     - a is a parameter that should be also described for the activity.
     - x is a state variable of the activity.
     - t is from the timeline of an activity.
    It is assumed that the time t runs from 0 to 1.

    Attributes:
         modelname(str): The name of the model which is used to describe the relation between the state and time. The
            following models are supported:
             - Linear: A linear relation between time and state. Parameters: xstart, xend
             - Spline3Knots: Two third order splines are used. Parameters: xstart, xend, a1, b1, c1, d1, a2, b2, c2, d2
    """
    def __init__(self, modelname):
        self.name = modelname

    def get_state(self, p, n=100):
        x = np.linspace(0, 1, n)
        if self.name == "Spline3Knots":
            x1 = x[:n // 2]
            x2 = x[n // 2:]
            y1 = p["a1"] * x1 ** 3 + p["b1"] * x1 ** 2 + p["c1"] * x1 + p["d1"]
            y2 = p["a2"] * x2 ** 3 + p["b2"] * x2 ** 2 + p["c2"] * x2 + p["d2"]
            y = np.concatenate((y1, y2))
            return p["xstart"] + y * (p["xend"] - p["xstart"])
        elif self.name == "Linear":
            return np.linspace(p["xstart"], p["xend"], n)
        else:
            raise ValueError('State cannot be computed for modelname "{:s}".'.format(self.name))

    def get_state_dot(self, p, n=100):
        x = np.linspace(0, 1, n)
        if self.name == "Spline3Knots":
            x1 = x[:n // 2]
            x2 = x[n // 2:]
            y1 = 3 * p["a1"] * x1 ** 2 + 2 * p["b1"] * x1 + p["c1"]
            y2 = 3 * p["a2"] * x2 ** 2 + 2 * p["b2"] * x2 + p["c2"]
            y = np.concatenate((y1, y2))
            return y * (p["xend"] - p["xstart"])
        elif self.name == "Linear":
            return np.ones(n) * (p["xend"] - p["xstart"])
        else:
            raise ValueError('State cannot be computed for modelname "{:s}".'.format(self.name))
