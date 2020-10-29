"""
Class Model


Author
------
Erwin de Gelder

Creation
--------
30 Oct 2018

To do
-----

Modifications
-------------
05 Nov 2018: Make code PEP8 compliant.
07 Nov 2018: Make seperate classes for each type of model. Model itself becomes abstract class.
12 Nov 2018: Add fit method to all models.
19 Nov 2018: Make it possible to instantiate models from JSON code.
14 Jan 2019: Add optional parameters tstart=0 and tend=2 to get_state and get_state_dor.
17 Jan 2019: @Joost Bosman: Fix options dictionary modified when using _set_default_options
             function in Model class.
13 Oct 2019: Update of terminology.
04 Nov 2019: Add constant model.
13 Mar 2020: BSplines added (since when was it removed?)
23 Mar 2020: MultiBSplines added.
26 Mar 2020: Fix bug when using different number of knots when using get_state().
27 Mar 2020: Export the initial model parameters when converting model to json.
29 Mar 2020: Enable the evaluation of the model at given time instants.
30 Apr 2020: Various fixes for BSplines model for some exceptions on the data.
2020 07 30: Update get_state_dot functionality, such that it is consistent with get_state().
"""

from abc import ABC, abstractmethod
import sys
from typing import List, Union
from scipy.interpolate import splev, splrep
from scipy.signal import lombscargle
from sklearn.metrics import mean_squared_error
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

    def get_state(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        """ Return state vector.

        The state is calculated based on the provided parameters. The time is
        assumed to be uniformly distributed. By default, 100 points are used to
        evaluate the state (to be altered using `npoints`).
        Alternatively, the time instances can be given using the `time`
        argument.

        :param pars: A dictionary with the parameters.
        :param npoints: Number of points for evaluating the state.
        :param time: Time instances at which the model is to be evaluated.
        :return: Numpy array with the state.
        """

    @staticmethod
    def _get_tdata(pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        """ Returning the time vector for _get_state.

        The time is assumed to be uniformly distributed. By default, 100 points
        are used to evaluate the state (to be altered using `npoints`).
        Alternatively, the time instances can be given using the `time`
        argument.

        :param pars: The parameters, should contain tstart and tend.
        :param npoints: Number of points for evaluating the state.
        :param time: Time instances at which the model is to be evaluated.
        :return: Time data.
        """
        if time is None:
            return np.linspace(0, 1, npoints)

        if isinstance(time, List):
            time = np.array(time)
        elif isinstance(time, float):
            time = np.array([time])
        return (time - pars["tstart"]) / (pars["tend"] - pars["tstart"])

    def get_state_dot(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        """ Return the derivative of the state vector.

        The state derivative is calculated based on the provided parameters.
        The time is assumed to be on the interval [0, 1], but can set
        differently using [tstart, tend]. By default, 100 points are used to
        evaluate the state (to be altered using npoints).

        :param pars: A dictionary with the parameters.
        :param npoints: Number of points for evaluating the state.
        :param time: Time instances at which the model is to be evaluated.
        :return: Numpy array with the derivative of the state.
        """

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

    The output is a constant value.
    """
    def __init__(self):
        Model.__init__(self, "Constant")

    def get_state(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        if time is None:
            return np.ones(npoints)*pars["xstart"]
        return np.ones(len(time))*pars["xstart"]

    def get_state_dot(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        return np.zeros(npoints)

    def fit(self, time: np.ndarray, data: np.ndarray, options: dict = None) -> dict:
        return dict(xstart=np.mean(data))


class Linear(Model):
    """ Linear model

    Linear relation between time and state. Parameters: xstart, xend.
    """
    def __init__(self):
        Model.__init__(self, "Linear")

    def get_state(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        time = self._get_tdata(pars, npoints, time)
        return pars["xstart"] + time*(pars["xend"] - pars["xstart"])

    def get_state_dot(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        return np.ones(npoints) * (pars["xend"] - pars["xstart"])

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
                    "xend": regression_result[0]*time_end + regression_result[1],
                    "tstart": time[0], "tend": time[-1]}
        if options["method"] == "endpoints":
            # Use the end points of the data to fit the linear line.
            index_begin = np.argmin(time)
            index_end = np.argmax(time)
            return {"xstart": data[index_begin], "xend": data[index_end],
                    "tstart": time[0], "tend": time[-1]}
        raise ValueError("Option '{}' for method is not valid.".format(options["method"]))


class Spline3Knots(Model):
    """ Spline model with 3 knots (one interior knot)

    Two third order splines are used.
    Parameters: xstart, xend, a1, b1, c1, d1, a2, b2, c2, d2.
    """
    def __init__(self):
        Model.__init__(self, "Spline3Knots")

    def get_state(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        tdata = self._get_tdata(pars, npoints, time)
        tdata1 = tdata[:npoints // 2]
        tdata2 = tdata[npoints // 2:]
        ydata1 = (pars["a1"] * tdata1 ** 3 + pars["b1"] * tdata1 ** 2 + pars["c1"] * tdata1 +
                  pars["d1"])
        ydata2 = (pars["a2"] * tdata2 ** 3 + pars["b2"] * tdata2 ** 2 + pars["c2"] * tdata2 +
                  pars["d2"])
        ydata = np.concatenate((ydata1, ydata2))
        return pars["xstart"] + ydata * (pars["xend"] - pars["xstart"])

    def get_state_dot(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        tdata = self._get_tdata(pars, npoints, time)
        tdata1 = tdata[:npoints // 2]
        tdata2 = tdata[npoints // 2:]
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
        parameters["tstart"] = time[0]
        parameters["tend"] = time[-1]
        return parameters


class Sinusoidal(Model):
    """ Sinusoidal model

    A sinusoidal model. Parameters: xstart, xend.
    """
    def __init__(self):
        Model.__init__(self, "Sinusoidal")

    def get_state(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        tdata = self._get_tdata(pars, npoints, time)
        offset = (pars["xstart"] + pars["xend"]) / 2
        amplitude = (pars["xstart"] - pars["xend"]) / 2
        return amplitude*np.cos(np.pi*tdata) + offset

    def get_state_dot(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        tdata = self._get_tdata(pars, npoints, time)
        amplitude = (pars["xstart"] - pars["xend"]) / 2
        return -np.pi*amplitude*np.sin(np.pi*tdata)

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
                "xend": lstlq_fit[1] - lstlq_fit[0],
                "tstart": time[0],
                "tend": time[-1]}


class BSplines(Model):
    """ Parametrise velocity profiles


    Example
    -------

    """
    TIME = 0
    VALUE = 1
    MIN_VALUE = 0
    MAX_VALUE = 1
    DEFAULT_PERIODOGRAM_FREQUENCIES = np.linspace(0.01, 2 * np.pi * 10, 100)

    def __init__(self, xdata: np.ndarray = None, ydata: np.ndarray = None, options: dict = None):
        """Initialise the class with the filename of the input file.

        Args
        ----
        xdata: Numpy array containing the x-VALUE to be parametrised.
        ydata: Numpy array containing the y-VALUE to be parametrised.
        options: Dictionary with all kinds of options.
        """

        Model.__init__(self, "BSplines")
        self.init_options = options

        # Define the default options
        self.default_options = {"n_knots": 4,
                                "degree": 3,
                                "ls_percentile": 95,
                                "frequencies": BSplines.DEFAULT_PERIODOGRAM_FREQUENCIES,
                                "n_random": 1000,
                                "min_n_knots": 4,
                                "max_n_knots": 12,
                                "uniform_knots": False,
                                "cut_off_freq": None,
                                "scaling": None,
                                "unique_mask": None,
                                "knot_positions": None,
                                "tck": None,
                                "lomb_scargle_boundary": None,
                                "residuals": None,
                                "keep_repetition_interval": 4}

        # Set the options
        self.options = Model._set_default_options(self, options)

        # Set data (if provided).
        self.data = None
        if xdata is not None and ydata is not None:
            self.load_data(xdata, ydata)

    def to_json(self):
        return dict(name="BSplines", init_parms=dict(options=self.init_options))

    def load_data(self, xdata: np.ndarray, ydata: np.ndarray) -> None:
        """ Load the data into a np.array

        Args
        ----
        xdata(np.ndarray): Numpy array containing the x-VALUE to be parametrised.
        ydata(np.ndarray): Numpy array containing the y-VALUE to be parametrised.
        """
        self._check_data(xdata, ydata)
        self.data = np.vstack([xdata, ydata])
        self._remove_duplicates()
        self._compute_scaling()
        self._scale_data()

    def _set_cut_off_frequency(self, cut_off_freq):
        """ Set the cut-off frequency"""

        self.options["cut_off_freq"] = cut_off_freq

    @staticmethod
    def _check_data(xdata, ydata):
        """ Check if the data has the correct format"""
        for data in [xdata, ydata]:
            if not isinstance(data, np.ndarray):
                raise TypeError(" Expected numpy array.")

            if len(data.shape) != 1:
                raise ValueError(" Arrays xdata and ydata should be one dimensional.")

        if not len(xdata) == len(ydata):
            raise ValueError(" Length of xdata should be the same as length of ydata.")

    def _remove_duplicates(self):
        """ Remove duplicates from the data

        Duplicates are possible if the data is logged at regular intervals but
        read out at different (larger) intervals
        """

        # False for consecutive VALUE that are unique, these should be removed
        mask = self.data[1] == np.roll(self.data[1], 1)
        sum_mask = np.zeros(len(mask))
        for i, entry in enumerate(mask[1:], start=1):
            if entry:
                sum_mask[i] = sum_mask[i - 1] + 1
                if sum_mask[i] == self.options["keep_repetition_interval"]:
                    sum_mask[i] = 0

        self.options["unique_mask"] = sum_mask == 0
        self.options["unique_mask"][-1] = True  # Always have the last value present.
        # self.data = self.data[:,mask]

    def _compute_scaling(self):
        """ Compute the scaling for scaling between 0 and 1
        Apply on both TIME and VALUE axes at the same time
        (by using axis=1 in the min and max function)"""
        self.options["scaling"] = np.transpose((np.min(self.data, axis=1),
                                                np.max(self.data, axis=1)))

        # If the values are constant, a small value is added to the maximum as to avoid a
        # divide-by-zero error.
        if self.options["scaling"][1][0] == self.options["scaling"][1][1]:
            self.options["scaling"][1][1] += 0.0001

    def _scale_data(self, this_data=None, axis=0):
        """ Scale the data between 0 and 1

        Args
        ----
        this_data : np.array(float)
            Data to be scaled between 0 and 1
        axis : int
            x-param: axis=Parametrisation.TIME
            y-param: axis=Parametrisation.VALUE

        When this_data is not given, self.data is scaled in both the
            TIME and VALUE direction
        """

        # In case no arguments are given the data of this instance is scaled
        if this_data is None:
            for i in (BSplines.TIME, BSplines.VALUE):
                self.data[i] = self._scale_data(self.data[i], i)
            return None
        # else scale given data
        return (this_data - self.options["scaling"][axis][BSplines.MIN_VALUE]) / \
               (self.options["scaling"][axis][BSplines.MAX_VALUE] -
                self.options["scaling"][axis][BSplines.MIN_VALUE])

    def _rescale_data(self, this_data, axis, order=0):
        """ Rescale the data

        Args
        ----
        this_data : np.array(float)
            Data between 0 and 1 to be rescaled
        axis : int
            x-param: axis=Parametrisation.TIME
            y-param: axis=Parametrisation.VALUE
        """
        factor = (self.options["scaling"][axis][BSplines.MAX_VALUE] -
                  self.options["scaling"][axis][BSplines.MIN_VALUE])
        offset = self.options["scaling"][axis][BSplines.MIN_VALUE]
        # Due to the chain rule the derivative needs to be scaled
        # both by time and value:
        #     g(x) = a f(cx + d) + b
        #     g'(x) = a/c f'(cx + d)
        if axis == BSplines.VALUE and order == 1:
            offset = 0
            factor /= (self.options["scaling"][BSplines.TIME][BSplines.MAX_VALUE] -
                       self.options["scaling"][BSplines.TIME][BSplines.MIN_VALUE])

        return offset + this_data * factor

    def _compute_knot_positions(self):
        """ Compute the position of the knots

        Two possibilities: free knots and fixed knots

        """

        if self.options["uniform_knots"]:
            # Place the knots at regular points
            time = self.data[0, self.options["unique_mask"]]
            self.options["knot_positions"] = \
                np.percentile(time, np.linspace(0, 100, self.options["n_knots"]+2))[1:-1]
        else:
            # Place the knots at regular intervals
            self.options["knot_positions"] = np.arange(1, self.options["n_knots"]+1) / \
                                  (self.options["n_knots"] + 1)

    def _compute_spline_representation(self):
        """ Compute the spline representation of the data given the knot
        positions

        """
        (time, value) = self.data[:, self.options["unique_mask"]]

        # Do a very quick and dirty solution in case we have too little data.
        if len(time) < 4:
            time = np.concatenate((np.linspace(time[0], time[1], 6 - len(time)), time[2:]))
            value = np.concatenate((np.linspace(value[0], value[1], 6-len(value)), value[2:]))

        self.options["tck"] = splrep(time, value, k=self.options["degree"],
                                     t=self.options["knot_positions"])

    def _compute_boundary(self, frequencies=None,
                          ls_percentile=None,
                          n_random=None):
        """ Compute lomb scargle boundary based on 'random noise'
            using the same time spacing as the original data.
            There will be n_random random noise patterns computed and
            for each pattern the max Lomb-Scargle periodogram value
            is collected. The pertencile value of the collected VALUE
            is returned

            Args
            ----
            frequencies : np.array(1,float)
                Numpy array containing the frequencies used in the
                Lomb-Scargle periodogram
                default value: BSplines.DEFAULT_PERIODOGRAM_FREQUENCIES
            ls_percentile : float, 0 <= value <= 100
                Percentile used for Lomb-Scargle max periodogram VALUE
                this is computed over all n_random generated random patterns
            n_random : int
                Number of random patterns to generate for the
                Lomb-Scargle boundary. More patterns will give
                a more accurate boundary but slower computation time
        """
        if frequencies is not None:
            self.options["frequencies"] = frequencies
        ls_percentile = ls_percentile if ls_percentile is not None\
            else self.default_options["ls_percentile"]
        n_random = n_random if n_random is not None\
            else self.default_options['n_random']

        max_per = []
        time = self.data[BSplines.TIME, self.options["unique_mask"]]
        for _ in range(0, n_random):
            per = lombscargle(time,
                              np.random.normal(size=len(time)),
                              frequencies,
                              normalize=True)
            max_per += [max(per)]
        self.options["lomb_scargle_boundary"] = np.percentile(max_per, ls_percentile)
        return self.options["lomb_scargle_boundary"]

    def _compute_residuals(self):
        """ Compute the residuals given the computed spline """
        (time, value) = self.data
        residual_value = value-splev(time, self.options["tck"])

        self.options["residuals"] = np.vstack([time, residual_value])

    # def _compute_autocorr(self):
    #    """ Compute the autocorrelation of the residuals """
    #    self.options["residuals"]

    def _compute_error_time(self, unique_datapoints=True):
        """ Compute the error of the spine representation in the time domain"""
        (time, value) = self.data[:, self.options["unique_mask"]] if\
            unique_datapoints is True else self.data

        return np.sqrt(mean_squared_error(splev(time, self.options["tck"]),
                                          value))

    def _compute_error_lomb_scargle(self, frequencies=None):
        """ Compute the Lomb-Scargle periodogram error
            of the spine representation in the frequency domain"""

        # array of frequencies for which to compute the periodogram
        if frequencies is not None:
            self.options["frequencies"] = frequencies

        (time, residuals) = self.options["residuals"][:, self.options["unique_mask"]]
        return max(lombscargle(time,
                               residuals,
                               self.options["frequencies"],
                               normalize=True))

    def _compute_error_frequency(self, cut_off_freq=None):
        """ Compute the error of the spline representation in the frequency
        domain

        """
        if cut_off_freq is not None:
            self.options["cut_off_freq"] = cut_off_freq
        (time, value) = self.data
        y_truth = abs(np.fft.rfft(value))
        y_pred = abs(np.fft.rfft(splev(time, self.options["tck"])))

        if self.options["cut_off_freq"] is not None:
            mask = np.arange(0, np.size(y_truth)) <= self.options["cut_off_freq"]
        else:
            mask = np.full(np.size(y_truth), True, dtype=bool)

        return np.sqrt(mean_squared_error(y_pred[mask], y_truth[mask]))

    def compute_error(self, frequencies=None,
                      unique_datapoints=True,
                      cut_off_freq=None):
        """ Compute the error in the time and frequency domain
            (fft and Lomb-Scargle)
            Args
            ----
            frequencies : np.array(1,float)
                Numpy array containing the frequencies used in the
                Lomb-Scargle periodogram
                default value: BSplines.DEFAULT_PERIODOGRAM_FREQUENCIES
            unique_datapoints: boolean
                True: Use only the unique datapoints for time error computation
                False: Use all datapoint for time error computation
           cut_off_freq : float
               Cut-off frequency in the error computation in the
               frequency domain
        """

        return (self._compute_error_time(unique_datapoints=unique_datapoints),
                self._compute_error_frequency(cut_off_freq=cut_off_freq),
                self._compute_error_lomb_scargle(frequencies=frequencies))

    def output_fit_error(self, cut_off_freq=None):
        """ Print fit error metrics
        """

        error = self.compute_error(cut_off_freq)

        print(' Fit error in the time domain: {0:1.3f}'.format(error[0]))

        if self.options["cut_off_freq"] is None:
            print(' Fit error in the frequency domain: '
                  '{0:1.3f}'.format(error[1]))

        else:
            print(' Fit error in the frequency domain for frequencies below '
                  '{0:}: {1:1.3f}'.format(self.options["cut_off_freq"], error[1]))

    def fit(self, time: np.ndarray = None, data: np.ndarray = None, options: dict = None):
        """ Parametrise the data using B-splines

        Args
        ----
        time: The x-VALUEs to be parametrised.
        data: The y-VALUEs to be parametrised.
        options: Dictionary with options for fit, e.g., degree, n_knots, uniform_knots.
        """

        if time is not None and data is not None:
            self.load_data(time, data)

        # Set the options. By default, use previously set options.
        self.default_options = self.options
        self.options = Model._set_default_options(self, options)

        self._compute_knot_positions()
        self._compute_spline_representation()
        self._compute_residuals()

        # Return the parameters
        coefficients, scaling, n_knots, knot_positions = self.get_spline_properties()
        if np.any(np.isnan(coefficients.tolist())):
            raise ValueError("NaN!!!")
        return {"coefficients": coefficients.tolist(),
                "scaling": scaling.tolist(),
                "n_knots": n_knots,
                "knot_positions": knot_positions.tolist()}

    def _try_n_knots(self, n_knots, uniform_knots=None):
        orig_n_knots = self.options["n_knots"]
        try:
            self.fit(options={"n_knots": n_knots, "uniform_knots": uniform_knots})
            return True
        except ValueError:
            print('Error: not insufficient points to fit all spline segments,'
                  'reverting to original #knots')
            # print(self.options["knot_positions"])
            # print(self.data[0,self.options["unique_mask"]])
            self.fit(options={"n_knots": orig_n_knots, "uniform_knots": uniform_knots})
            return False

    def determine_n_knots(self, options: dict = None):
        """ Compute optimal number of knots based on Lomb-Scargle
            periodogram boundary
        Args
        ----
        options: Dictionary with options for determining the
        number of interior knots.These options include:
            min_n_knots: int
                Minimal number of interior knots used in the spline fit
            max_n_knots: int
                Maximal number of interior knots used in the spline fit
            ls_percentile : float, 0 <= value <= 100
                    Percentile used for Lomb-Scargle max periodogram VALUE
                    this is computed over all n_random generated random patterns
            uniform_knots: boolean
                True: Distribute interior knots evenly over datapoints
                False: Distribute interior knots evenly over time
            n_random : int
                    Number of random patterns to generate for the
                    Lomb-Scargle boundar. More patterns will give a
                    more accurate boundary but slower computation time.
        """
        # Set the options. By default, use previously set options.
        self.default_options = self.options
        self.options = Model._set_default_options(self, options)

        min_n_knots = self.options["min_n_knots"]
        max_n_knots = self.options["max_n_knots"]
        frequencies = self.options["frequencies"]
        ls_percentile = self.options["ls_percentile"]
        n_random = self.options["n_random"]
        uniform_knots = self.options["uniform_knots"]

        lombscargle_boundary = self._compute_boundary(frequencies=frequencies,
                                                      ls_percentile=ls_percentile,
                                                      n_random=n_random)

        succes = self._try_n_knots(n_knots=min_n_knots,
                                   uniform_knots=uniform_knots)
        max_lombscargle =\
            self._compute_error_lomb_scargle(frequencies=frequencies)
        while succes\
            and max_lombscargle > lombscargle_boundary\
                and self.options["n_knots"] < max_n_knots:
            succes = self._try_n_knots(n_knots=self.options["n_knots"]+1,
                                       uniform_knots=uniform_knots)
            max_lombscargle =\
                self._compute_error_lomb_scargle(frequencies=frequencies)

        return self.options["n_knots"], lombscargle_boundary

    def predict(self, x_values: np.ndarray, order=0) -> np.ndarray:
        """ Evaluate the spline representation

        If x is given, spline is evaluated at x, otherwise at self.data[0].

        Args
        ----
        x_values : Array with VALUE between 0 and 1 at which the spline fit is evaluated
        order : Default 0 (original fit). Computes the order derivative.
                order should be less or equal to the spline degree
        """

        if not isinstance(x_values, np.ndarray):
            raise TypeError(' Expected numpy array')

        x_scaled = self._scale_data(x_values.copy(), BSplines.TIME)
        return self._predict_unscaled(x_scaled, order)

    def _predict_unscaled(self, x_values: np.ndarray, order=0) -> np.ndarray:
        if not isinstance(x_values, np.ndarray):
            raise TypeError(' Expected numpy array')

        return self._rescale_data(splev(x_values, self.options["tck"], order),
                                  BSplines.VALUE, order)

    def get_state(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        self.set_spline_properties(pars["coefficients"], pars["scaling"],
                                   pars["n_knots"], pars["knot_positions"])
        xdata = self._get_tdata(dict(tstart=pars["scaling"][0][0], tend=pars["scaling"][0][1]),
                                npoints, time)
        return self._predict_unscaled(xdata)

    def get_state_dot(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        self.set_spline_properties(pars["coefficients"], pars["scaling"],
                                   pars["n_knots"], pars["knot_positions"])
        xdata = self._get_tdata(dict(tstart=pars["scaling"][0][0], tend=pars["scaling"][0][1]),
                                npoints, time)
        return self._predict_unscaled(xdata, order=1)

    def get_parameter(self, key):
        """ Request a parameter value

        Args
        ----
        key: The key of the requested parameter value

        Returns
        -------
        The requested parameter corresponding to the "key" argument

        """
        return self.options[key]

    def get_spline_properties(self):
        """ Return the spline properties

        Returns
        -------
        Coefficients: Numpy array with (1 + k + n_knots) values
        Scaling: Numpy array of shape(2,2):
            array([[x_min,x_max],[y_min, y_max]])
        n_knots: 1 value
        interior_positions: uniform: Empty Numpy array
                            otherwise: Numpy array of length (n_knots)
        k can be derived as follows: k=#coefficients - n_knots - 1
        """

        # Get spline coefficients
        knot_positions, coefficients, degree = self.options["tck"]
        if self.options["uniform_knots"]:
            # Store interior knot positions
            knot_positions = self.options["knot_positions"]
        else:
            # For evenly spaced knots the knot positions are not required
            # as these can be reconstructed from the other parameters
            knot_positions = np.array([])
        # Remove padded coefficient values added by the splrep function
        coefficients = coefficients[:-degree-1]

        return coefficients, self.options["scaling"], self.options["n_knots"], knot_positions

    def set_spline_properties(self, coefficients, scaling, n_knots, knot_positions=None):
        """ Set the spline properties

        Args
        ----
        coefficients: Numpy array with (1 + k + n_knots) values
        scaling: Numpy array of shape(2,2):
            array([[x_min,x_max],[y_min, y_max]])
        n_knots: 1 value
        knot_positions: uniform: Empty Numpy array
                        otherwise: Numpy array of length (n_knots)
        k will be derived as follows: k=#coefficients - n_knots - 1
        """

        self.options["n_knots"] = n_knots
        if knot_positions is not None and knot_positions:
            self.options["uniform_knots"] = True
            self.options["knot_positions"] = knot_positions
        else:
            # Compute knot positions from other parameters
            self.options["uniform_knots"] = False
            self._compute_knot_positions()
        degree = len(coefficients) - n_knots - 1
        knot_positions = np.concatenate((
            np.zeros(degree+1),
            self.options["knot_positions"],
            np.ones(degree+1)))
        coefficients = np.concatenate((coefficients, np.zeros(degree+1)))
        self.options["scaling"] = scaling
        self.options["degree"] = degree
        self.options["tck"] = (knot_positions, coefficients, degree)


class MultiBSplines(Model):
    """ BSplines, dealing with multivariate data. """
    def __init__(self, dimension: int, options: dict = None):
        """Initialise the class with the filename of the input file.

        Args
        ----
        dimension: The dimension of the data.
        options: Dictionary with all kinds of options.
        """
        Model.__init__(self, "MultiBSplines")

        # Initialize the b-splines.
        self.dimension = dimension
        self.options = options
        self.bsplines = [BSplines(options=self.options) for _ in range(dimension)]

    def fit(self, time: np.ndarray, data: np.ndarray, options: dict = None):
        # Set data correctly.
        n_data = len(time)
        if data.shape == (n_data, self.dimension):
            data = data.T
        elif not data.shape == (self.dimension, n_data):
            raise ValueError("Data should be n-by-d or d-by-n, where d is the provided dimension.")

        # Loop through the different dimensions.
        all_pars = [bspline.fit(time, data[i], options) for i, bspline in enumerate(self.bsplines)]
        pars = dict(coefficients=[par['coefficients'] for par in all_pars],
                    scaling=[par['scaling'] for par in all_pars],
                    n_knots=[par["n_knots"] for par in all_pars],
                    knot_positions=[par["knot_positions"] for par in all_pars])

        return pars

    def get_state(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        result = np.array([bspline.get_state(dict(coefficients=pars["coefficients"][i],
                                                  scaling=pars["scaling"][i],
                                                  n_knots=pars["n_knots"][i],
                                                  knot_positions=pars["knot_positions"][i]),
                                             npoints, time)
                           for i, bspline in enumerate(self.bsplines)])
        return result

    def get_state_dot(self, pars: dict, npoints: int = 100, time: np.ndarray = None) -> np.ndarray:
        result = np.array([bspline.get_state_dot(dict(coefficients=pars["coefficients"][i],
                                                      scaling=pars["scaling"][i],
                                                      n_knots=pars["n_knots"][i],
                                                      knot_positions=pars["knot_positions"][i]),
                                                 npoints, time)
                           for i, bspline in enumerate(self.bsplines)])
        return result

    def to_json(self):
        return dict(name="MultiBSplines", init_parms=dict(dimension=self.dimension,
                                                          options=self.options))


def model_from_json(json: Union[str, dict]) -> Model:
    """ Get Model object from JSON code

    It is assumed that the JSON code of the Model is created using
    Model.to_json().

    :param json: JSON code of Model, which is simply a string of the name of the
        Model or a dictionary that also contains the options.
    :return: Model object.
    """
    if isinstance(json, str):
        return getattr(sys.modules[__name__], json)()
    return getattr(sys.modules[__name__], json["name"])(**json["init_parms"])
