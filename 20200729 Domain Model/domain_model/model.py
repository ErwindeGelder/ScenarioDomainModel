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
"""

import sys
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
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
        name(str): The name of the model which is used to describe the relation
            between the state and time.
        default_options(dict): Dictionary with the default options that are used
            for fitting data to the model.
    """
    @abstractmethod
    def __init__(self, modelname: str):
        self.name = modelname
        self.default_options = dict()

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
    def fit(self, time: np.ndarray, data: np.ndarray, options: dict = None) -> dict:
        """ Fit the data to the model and return the parameters

        The data is to be fit to the model and the resulting parameters are
        returned using a dictionary. The input data needs to be a n-by-m array,
        where n denotes the number of datapoints and each datapoint has
        dimension m. The time should be an vector with length of n.

        :param time: the time instants of the data.
        :param data: the data that will be fit to the model.
        :param options: specify some model-specific options.
        :return: dictionary of the parameters.
        """

    def to_json(self):
        """ Function that can be called when exporting Model to JSON.

        Currently, only the name of the model is returned. This might change
        later. For example, a short description (e.g., the formula) might be
        given as well.
        """
        return self.name

    def _set_default_options(self, options: dict = None) -> dict:
        if options is None:
            options = {}
        else:
            # Make local copy in order to prevent changes to original.
            options = options.copy()

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
    def __init__(self):
        Model.__init__(self, "Constant")

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return np.ones(len(time))*pars["xstart"]

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return np.zeros(len(time))

    def fit(self, time: np.ndarray, data: np.ndarray, options: dict = None) -> dict:
        return dict(xstart=np.mean(data))


class Linear(Model):
    """ Linear model

    Linear relation between time and state. Parameters: xstart, xend.
    """
    def __init__(self):
        Model.__init__(self, "Linear")

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return pars["xstart"] + time*(pars["xend"] - pars["xstart"])

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        return np.ones(len(time)) * (pars["xend"] - pars["xstart"])

    def fit(self, time: np.ndarray, data: np.ndarray, options: dict = None) -> dict:
        """ Fit the data to the model and return the parameters.

        The data is to be fit to the model and the resulting parameters are
        returned in a dictionary. The input data needs to be a n-by-m array,
        where n denotes the number of datapoints and each datapoint has
        dimension m. The time should be an vector with length of n. For this
        model (Linear), only m=1 is supported.

        As options, the "method" can be passed. Two different methods are
        possible:
         - least_squares: make a least squares fit.
         - endpoints: Only the starting point and the end point are used to fit
           the model.

        :param time: the time instants of the data.
        :param data: the data that will be fit to the model.
        :param options: specify some model-specific options.
        :return: dictionary of the parameters.
        """
        # Set the options correctly
        self.default_options["method"] = "least_squares"
        options = Model._set_default_options(self, options)

        if options["method"] == "least_squares":
            # Use least squares regression to find the slope of the linear line.
            matrix = np.array([time, np.ones(len(time))]).T
            regression_result = np.linalg.lstsq(matrix, data)[0]
            time_begin = np.min(time)
            time_end = np.max(time)
            return {"xstart": regression_result[0]*time_begin + regression_result[1],
                    "xend": regression_result[0]*time_end + regression_result[1]}
        if options["method"] == "endpoints":
            # Use the end points of the data to fit the linear line.
            index_begin = np.argmin(time)
            index_end = np.argmax(time)
            return {"xstart": data[index_begin], "xend": data[index_end]}
        raise ValueError("Option '{}' for method is not valid.".format(options["method"]))


class Spline3Knots(Model):
    """ Spline model with 3 knots (one interior knot)

    Two third order splines are used.
    Parameters: xstart, xend, a1, b1, c1, d1, a2, b2, c2, d2.
    """
    def __init__(self):
        Model.__init__(self, "Spline3Knots")

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        tdata1 = time[time < .5]
        tdata2 = time[time >= .5]
        ydata1 = (pars["a1"] * tdata1 ** 3 + pars["b1"] * tdata1 ** 2 + pars["c1"] * tdata1 +
                  pars["d1"])
        ydata2 = (pars["a2"] * tdata2 ** 3 + pars["b2"] * tdata2 ** 2 + pars["c2"] * tdata2 +
                  pars["d2"])
        ydata = np.concatenate((ydata1, ydata2))
        return pars["xstart"] + ydata * (pars["xend"] - pars["xstart"])

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        tdata1 = time[time < .5]
        tdata2 = time[time >= .5]
        ydata1 = 3 * pars["a1"] * tdata1 ** 2 + 2 * pars["b1"] * tdata1 + pars["c1"]
        ydata2 = 3 * pars["a2"] * tdata2 ** 2 + 2 * pars["b2"] * tdata2 + pars["c2"]
        ydata = np.concatenate((ydata1, ydata2))
        return ydata * (pars["xend"] - pars["xstart"])

    def fit(self, time: np.ndarray, data: np.ndarray, options: dict = None) -> dict:
        """ Fit the data to the model and return the parameters.

        The data is to be fit to the model and the resulting parameters are
        returned in a dictionary. The input data needs to be a n-by-m array,
        where n denotes the number of datapoints and each datapoint has
        dimension m. The time should be an vector with length of n. For this
        model (Spline3Knots), only m=1 is supported.

        :param time: the time instants of the data.
        :param data: the data that will be fit to the model.
        :param options: specify some model-specific options.
        :return: dictionary of the parameters.
        """
        # Normalize the time
        time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))

        # Scale the data
        parameters = {"xstart": np.max(data),
                      "xend": np.min(data)}
        scaled_data = (data - parameters["xstart"]) / (parameters["xend"] - parameters["xstart"])

        # Create the matrix that will be used for the least squares regression
        matrix = np.array([time_normalized**3, time_normalized**2, time_normalized**1,
                           np.ones(len(time_normalized))]).T
        matrix_left_spline = matrix.copy()
        matrix_left_spline[time >= 0.5] = 0
        matrix_right_spline = matrix.copy()
        matrix_right_spline[time < 0.5] = 0
        matrix = np.concatenate((matrix_left_spline, matrix_right_spline), axis=1)

        # Construct the constraint matrix, 3 constraints, 8 coefficients
        constraint_matrix = np.array([[1, 1, 1, 1, -1, -1, -1, -1],
                                      [3, 2, 1, 0, -3, -2, -1, 0],
                                      [6, 2, 0, 0, -6, -2, 0, 0]])
        # And create the nullspace which is used for fitting the model
        nullspace_constraint_matrix = np.linalg.svd(constraint_matrix)[2][3:]

        lstsq_fit = np.linalg.lstsq(np.dot(matrix, nullspace_constraint_matrix.T), scaled_data,
                                    rcond=None)[0]
        theta = np.dot(nullspace_constraint_matrix.T, lstsq_fit)
        parameters["a1"] = theta[0]
        parameters["b1"] = theta[1]
        parameters["c1"] = theta[2]
        parameters["d1"] = theta[3]
        parameters["a2"] = theta[4]
        parameters["b2"] = theta[5]
        parameters["c2"] = theta[6]
        parameters["d2"] = theta[7]
        return parameters


class Sinusoidal(Model):
    """ Sinusoidal model

    A sinusoidal model. Parameters: xstart, xend.
    """
    def __init__(self):
        Model.__init__(self, "Sinusoidal")

    def get_state(self, pars: dict, time: np.ndarray) -> np.ndarray:
        offset = (pars["xstart"] + pars["xend"]) / 2
        amplitude = (pars["xstart"] - pars["xend"]) / 2
        return amplitude*np.cos(np.pi*time) + offset

    def get_state_dot(self, pars: dict, time: np.ndarray) -> np.ndarray:
        amplitude = (pars["xstart"] - pars["xend"]) / 2
        return -np.pi*amplitude*np.sin(np.pi*time)

    def fit(self, time: np.ndarray, data: np.ndarray, options: dict = None) -> dict:
        """ Fit the data to the model and return the parameters.

        The data is to be fit to the model and the resulting parameters are
        returned in a dictionary. The input data needs to be a n-by-m array,
        where n denotes the number of datapoints and each datapoint has
        dimension m. The time should be an vector with length of n. For this
        model (Sinusoidal), only m=1 is supported.

        :param time: the time instants of the data.
        :param data: the data that will be fit to the model.
        :param options: specify some model-specific options.
        :return: dictionary of the parameters.
        """
        # Normalize the time
        time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time))

        # Use least squares regression to find the amplitude and the offset
        matrix = np.array([np.cos(np.pi*time_normalized), np.ones(len(time))]).T
        lstlq_fit = np.linalg.lstsq(matrix, data, rcond=None)[0]

        # Return the parameters
        return {"xstart": lstlq_fit[0] + lstlq_fit[1],
                "xend": lstlq_fit[1] - lstlq_fit[0]}


def model_from_json(json: str) -> Model:
    """ Get Model object from JSON code

    It is assumed that the JSON code of the Model is created using
    Model.to_json().

    :param json: JSON code of Model, which is simply a string of the name of the
        Model.
    :return: Model object.
    """
    return getattr(sys.modules[__name__], json)()
