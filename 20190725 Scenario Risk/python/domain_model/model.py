""" Class Model

Creation date: 2018 10 30
Author(s): Erwin de Gelder

Modifications:
2018 11 05: Make code PEP8 compliant.
2018 11 07: Make seperate classes for each type of model. Model itself becomes abstract class.
2018 11 12: Add fit method to all models.
2018 11 19: Make it possible to instantiate models from JSON code.
2019 01 14: Add optional parameters tstart=0 and tend=2 to get_state and get_state_dor.
2019 10 13: Update of terminology.
2019 11 04: Add constant model.
2020 08 15: Make sure that each model has the functions `get_state`, `get_state_dot`, and `fit`.
2020 08 23: Enable the evaluation of the model at given time instants.
2020 08 24: Spline model added.
2020 10 02: Make Model a subclass of QualitativeElement.
2020 10 04: Change way of creating object from JSON code.
2020 10 30: For using options for the fit functions, use **kwargs instead of options.
2020 11 06: Add Model MultiBSplines.
"""

import sys
from abc import abstractmethod
import numpy as np
from scipy.interpolate import splrep, splev
from .qualitative_element import QualitativeElement, _qualitative_element_props_from_json
from .scenario_element import DMObjects, _object_from_json


class Model(QualitativeElement):
    """ Model

    Parameter Model describes the relation between the states variables and the
    parameters that specify an activity.

    Example :
    x = a * t

    In this case,
     - a is a parameter that should be also described for the activity.
     - x is a state variable of the activity.
     - t is from the timeline of an activity.
    It is assumed that the time t runs from 0 to 1.

    Attributes:
        modelname(str): The name of the model which is used to describe the
            relation between the state and time.
        default_options(dict): Dictionary with the default options that are used
            for fitting data to the model.
        uid (int): A unique ID.
        name (str): A name that serves as a short description of the actor
            category.
        tags (List[Tag]): The tags are used to determine whether a scenario
            category comprises a scenario.
        description(str): A string that qualitatively describes this thing.
    """
    @abstractmethod
    def __init__(self, modelname: str, **kwargs):
        self._modelname = modelname
        self.default_options = dict()
        QualitativeElement.__init__(self, **kwargs)

    @abstractmethod
    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        """ Return state vector.

        The state is calculated based on the provided parameters. The default
        time of the model is always on the interval [0, 1], so any time values
        outside this interval can be regarded as extrapolation.

        :param pars: A dictionary with the parameters.
        :param time: Time instances at which the model is to be evaluated.
        :return: Numpy array with the state.
        """

    @abstractmethod
    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        """ Return the derivative of the state vector.

        The state derivative is calculated based on the provided parameters.
        The default time of the model is always on the interval [0, 1], so any
        time values outside this interval can be regarded as extrapolation.

        :param pars: A dictionary with the parameters.
        :param time: Time instances at which the model is to be evaluated.
        :return: Numpy array with the derivative of the state.
        """

    @abstractmethod
    def fit(self, time: np.ndarray, data: np.ndarray, **kwargs) -> dict:
        """ Fit the data to the model and return the parameters

        The data is to be fit to the model and the resulting parameters are
        returned using a dictionary. The input data needs to be a n-by-m array,
        where n denotes the number of datapoints and each datapoint has
        dimension m. The time should be an vector with length of n.

        :param time: the time instants of the data.
        :param data: the data that will be fit to the model.
        :param kwargs: specify some model-specific options.
        :return: dictionary of the parameters.
        """

    def to_json(self) -> dict:
        model = QualitativeElement.to_json(self)
        model["modelname"] = self._modelname
        model["default_options"] = self.default_options
        return model

    def _set_default_options(self, **kwargs) -> dict:
        # Make local copy in order to prevent changes to original.
        options = kwargs.copy()

        # Check for options that are not set by default --> this is an invalid options.
        for option in options:
            if option not in self.default_options:
                raise ValueError("Option '{:s}' is not a valid options.".format(option))

        # Loop through the default options. If options is already set, then ignore it. If
        # options is not already set, then use the default options.
        for key, value in self.default_options.items():
            if key not in options.keys():
                options[key] = value
        return options


class Constant(Model):
    """ Constant model

    The output is a constant value. Parameters: xstart.
    """
    def __init__(self, **kwargs):
        Model.__init__(self, "Constant", **kwargs)

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return np.ones(len(time))*pars["xstart"]

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return np.zeros(len(time))

    def fit(self, time: np.ndarray, data: np.ndarray, **kwargs) -> dict:
        return dict(xstart=np.mean(data))


class Linear(Model):
    """ Linear model

    Linear relation between time and state. Parameters: xstart, xend.

    As options for fitting, the "method" can be passed. Two different methods
    are possible:
     - least_squares: make a least squares fit (default).
     - endpoints: Only the starting point and the end point are used to fit
       the model.
    """
    def __init__(self, endpoints=False, **kwargs):
        Model.__init__(self, "Linear", **kwargs)
        self.default_options = dict(endpoints=endpoints)

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return pars["xstart"] + time*(pars["xend"] - pars["xstart"])

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return np.ones(len(time)) * (pars["xend"] - pars["xstart"])

    def fit(self, time: np.ndarray, data: np.ndarray, **kwargs) -> dict:
        # Set the options correctly
        options = Model._set_default_options(self, **kwargs)

        if not options["endpoints"]:
            # Use least squares regression to find the slope of the linear line.
            matrix = np.array([time, np.ones(len(time))]).T
            regression_result = np.linalg.lstsq(matrix, data, rcond=None)[0]
            time_begin = np.min(time)
            time_end = np.max(time)
            return {"xstart": regression_result[0]*time_begin + regression_result[1],
                    "xend": regression_result[0]*time_end + regression_result[1]}

        # Use the end points of the data to fit the linear line.
        index_begin = np.argmin(time)
        index_end = np.argmax(time)
        return {"xstart": data[index_begin], "xend": data[index_end]}


class Sinusoidal(Model):
    """ Sinusoidal model

    A sinusoidal model. Parameters: xstart, xend.
    """
    def __init__(self, **kwargs):
        Model.__init__(self, "Sinusoidal", **kwargs)

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        offset = (pars["xstart"] + pars["xend"]) / 2
        amplitude = (pars["xstart"] - pars["xend"]) / 2
        return amplitude*np.cos(np.pi*time) + offset

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        amplitude = (pars["xstart"] - pars["xend"]) / 2
        return -np.pi*amplitude*np.sin(np.pi*time)

    def fit(self, time: np.ndarray, data: np.ndarray, **kwargs) -> dict:
        # Normalize the time
        time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))

        # Use least squares regression to find the amplitude and the offset
        matrix = np.array([np.cos(np.pi*time_normalized), np.ones(len(time))]).T
        lstlq_fit = np.linalg.lstsq(matrix, data, rcond=None)[0]

        # Return the parameters
        return {"xstart": lstlq_fit[0] + lstlq_fit[1],
                "xend": lstlq_fit[1] - lstlq_fit[0]}


class Spline3Knots(Model):
    """ Spline model with 3 knots (one interior knot)

    Two third order splines are used.
    Parameters: a1, b1, c1, d1, a2, b2, c2, d2.

    For fitting, the options `endpoints` (default: False) can be set to True if
    the spline function should ensure that the start and end values are the same
    as for the provided data.
    """
    def __init__(self, endpoints=False, **kwargs):
        Model.__init__(self, "Spline3Knots", **kwargs)
        self.default_options = dict(endpoints=endpoints)

        self.constraint_matrix = np.array([[1, 2, 4, 8, -1, -2, -4, -8],
                                           [3, 4, 4, 0, -3, -4, -4, 0],
                                           [3, 2, 0, 0, -3, -2, 0, 0]])
        self.constraint_matrix_endpoints = np.concatenate((self.constraint_matrix,
                                                           [[0, 0, 0, 1, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 1, 1, 1, 1]]))
        self.vh_matrix = np.linalg.svd(self.constraint_matrix)[2]
        self.usvh_endpoints = np.linalg.svd(self.constraint_matrix_endpoints)

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        tdata1 = time[time < .5]
        tdata2 = time[time >= .5]
        ydata1 = (pars["a1"]*tdata1**3 + pars["b1"]*tdata1**2 + pars["c1"]*tdata1 + pars["d1"])
        ydata2 = (pars["a2"]*tdata2**3 + pars["b2"]*tdata2**2 + pars["c2"]*tdata2 + pars["d2"])
        return np.concatenate((ydata1, ydata2))

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        tdata1 = time[time < .5]
        tdata2 = time[time >= .5]
        ydata1 = 3*pars["a1"]*tdata1**2 + 2*pars["b1"]*tdata1 + pars["c1"]
        ydata2 = 3*pars["a2"]*tdata2**2 + 2*pars["b2"]*tdata2 + pars["c2"]
        return np.concatenate((ydata1, ydata2))

    def fit(self, time: np.ndarray, data: np.ndarray, **kwargs) -> dict:
        options = self._set_default_options(**kwargs)

        # Normalize the time
        time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))

        # Create the matrix that will be used for the least squares regression
        matrix = np.array([time_normalized**3, time_normalized**2, time_normalized**1,
                           np.ones(len(time_normalized))]).T
        matrix_left_spline = matrix.copy()
        matrix_left_spline[time_normalized >= 0.5] = 0
        matrix_right_spline = matrix.copy()
        matrix_right_spline[time_normalized < 0.5] = 0
        matrix = np.concatenate((matrix_left_spline, matrix_right_spline), axis=1)

        # Construct the constraint matrix, 3 constraints, 8 coefficients
        if options["endpoints"]:
            theta_default = np.dot(self.usvh_endpoints[2][:5].T,
                                   np.dot(self.usvh_endpoints[0].T,
                                          np.array([0, 0, 0, data[0], data[-1]])) /
                                   self.usvh_endpoints[1])
            theta_fit = np.linalg.lstsq(np.dot(matrix, self.usvh_endpoints[2][5:].T),
                                        data - np.dot(matrix, theta_default), rcond=None)[0]
            theta = np.dot(self.usvh_endpoints[2][5:].T, theta_fit) + theta_default
        else:
            theta_fit = np.linalg.lstsq(np.dot(matrix, self.vh_matrix[3:].T), data,
                                        rcond=None)[0]
            theta = np.dot(self.vh_matrix[3:].T, theta_fit)

        return dict(a1=theta[0], b1=theta[1], c1=theta[2], d1=theta[3],
                    a2=theta[4], b2=theta[5], c2=theta[6], d2=theta[7])


class Splines(Model):
    """ Spline model with a variable number of knots.

    Model using the spline functionality of scipy's splrep and splev.
    Parameters: knots, coefficients, degree

    When using the fit-function, the following options can be used:
    - degree: the degree of the splines (default=3).
    - n_knots: the number of interior knots (default=3).
    The interior knots will be evenly distributed.
    """
    def __init__(self, degree=3, n_knots=3, **kwargs):
        Model.__init__(self, "Splines", **kwargs)
        self.default_options = dict(degree=degree, n_knots=n_knots)

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return splev(time, (pars["knots"], pars["coefficients"], pars["degree"]))

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return splev(time, (pars["knots"], pars["coefficients"], pars["degree"]), 1)

    def fit(self, time: np.ndarray, data: np.ndarray, **kwargs) -> dict:
        # Normalize the time
        time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))

        # Set options.
        options = self._set_default_options(**kwargs)

        # Set interior knots.
        knots = np.arange(1, options["n_knots"]+1) / (options["n_knots"] + 1)

        # Compute spline coefficients.
        tck = splrep(time_normalized, data, k=options["degree"], t=knots)
        pars = dict(knots=tck[0].tolist(), coefficients=tck[1].tolist(), degree=tck[2])
        return pars


class MultiBSplines(Model):
    """ BSplines, dealing with multivariate data. """
    def __init__(self, dimension: int, degree=3, n_knots=3, **kwargs):
        Model.__init__(self, "MultiBSplines", **kwargs)

        # Initialize the b-splines.
        self.dimension = dimension
        self.default_options = dict(dimension=dimension, degree=degree, n_knots=n_knots)
        self.spline = Splines(degree=degree, n_knots=n_knots)

    def fit(self, time: np.ndarray, data: np.ndarray, **kwargs):
        # Set data correctly.
        n_data = len(time)
        if data.shape == (n_data, self.dimension):
            data = data.T
        elif not data.shape == (self.dimension, n_data):
            raise ValueError("Data should be n-by-d or d-by-n, where d is the provided dimension.")

        # Loop through the different dimensions.
        all_pars = [self.spline.fit(time, data[i], **kwargs) for i in range(self.dimension)]
        pars = dict(coefficients=[par['coefficients'] for par in all_pars],
                    knots=[par["knots"] for par in all_pars],
                    degree=[par["degree"] for par in all_pars])

        return pars

    def get_state(self, pars: dict, time: np.ndarray = None) -> np.ndarray:
        result = np.array([self.spline.get_state(dict(coefficients=pars["coefficients"][i],
                                                      degree=pars["degree"][i],
                                                      knots=pars["knots"][i]),
                                                 time)
                           for i in range(self.dimension)])
        return result

    def get_state_dot(self, pars: dict, time: np.ndarray = None) -> np.ndarray:
        result = np.array([self.spline.get_state_dot(dict(coefficients=pars["coefficients"][i],
                                                          degree=pars["degree"][i],
                                                          knots=pars["knots"][i]),
                                                     time)
                           for i in range(self.dimension)])
        return result


def _model_props_from_json(json: dict) -> dict:
    props = json["default_options"]
    props.update(_qualitative_element_props_from_json(json))
    return props


def _model_from_json(
        json: dict,
        attribute_objects: DMObjects  # pylint: disable=unused-argument
) -> Model:
    return getattr(sys.modules[__name__], json["modelname"])(**_model_props_from_json(json))


def model_from_json(json: dict, attribute_objects: DMObjects = None) -> Model:
    """ Get Model object from JSON code

    It is assumed that the JSON code of the Model is created using
    Model.to_json().

    :param json: JSON code of Model.
    :param attribute_objects: A structure for storing all objects (optional).
    :return: Model object.
    """
    return _object_from_json(json, _model_from_json, "model", attribute_objects)
