""" Class for the detection of change points.

Creation date: 2020 02 28
Author(s): Erwin de Gelder

Modifications:
"""


from typing import Callable, List, NamedTuple, Tuple
import numpy as np
from options import Options


class ChangePointOptions(Options):
    """ Options for the change point detector. """
    min_length: int = 10  # Minimum sample length between two consecutive change points.
    stop: float = 0.1  # A number between 0 and 1 that defines when the algorithm has to stop.
    plot_flag: bool = False  # Whether to show the results in a plot.


Candidate = NamedTuple("candidate", (("index", int), ("error", float), ("time", float),
                                     ("coefficients1", np.ndarray), ("coefficients2", np.ndarray)))


class ChangePointDetector:
    """ Detect change points.

    The method for the detection of change points is mostly based on the paper
    'Event Detection from Time Series Data' from Guralnik and Srivastava (1999).

    Attributes:
        options(ChangePointOptions): Configuration parameters
    """
    def __init__(self, functions: List[Callable], xdata: np.ndarray, ydata: np.ndarray,
                 options: ChangePointOptions = None):
        self.functions = functions
        self.xdata = xdata
        if len(ydata.shape) == 1:
            self.ydata = ydata[:, np.newaxis]
        elif len(ydata.shape) == 2:
            self.ydata = ydata
        else:
            raise ValueError("ydata needs to be either one dimensional or two dimensional.")
        self.options = ChangePointOptions() if options is None else options

    def find_min_risk(self, xdata: np.array, ydata: np.ndarray) -> Tuple[float, np.ndarray]:
        """ Compute the minimum risk.

        The minimum risk is defined as the sum of squared errors, where the
        error is the difference between y and the best fit. The best fit is
        found by using the linear least squares method.

        :param xdata: The time instants of the considered data.
        :param ydata: The measured data at the given time instants.
        :return: The minimum risk.
        """
        values = np.zeros((len(xdata), len(self.functions)))
        for i, function in enumerate(self.functions):
            values[:, i] = function(xdata)
        coefficients, residuals, _, _ = np.linalg.lstsq(values, ydata, rcond=None)
        return float(np.sum(residuals)), np.array(coefficients)

    def find_candidate(self, xdata: np.ndarray, ydata: np.ndarray) -> Candidate:
        """ Finds the optimal change point and the corresponding coefficients.

        The functions returns:
         - index of the instant at which the change point occurs
         - error is a 2-by-1 vector with the errors of the first section (i.e.
           before the change point) and the second section (i.e. after the
           change point). See find_min_risk for the definition of the error
         - the coefficients for the first section
         - the coefficients for the second section

        :param xdata: The time instants of the considered data.
        :param ydata: The measured data at the given time instants.
        :return: Information on the candidate change point.
        """
        # Loop through all possible change points and compute the error.
        min_index = 0
        min_error = np.inf
        best_coefficients = np.array([]), np.array([])
        for i in range(self.options.min_length, len(xdata)-self.options.min_length+1):
            error1, coefficients1 = self.find_min_risk(xdata[:i], ydata[:i])
            error2, coefficients2 = self.find_min_risk(xdata[i:], ydata[i:])
            if error1+error2 < min_error:
                min_index = i
                min_error = error1 + error2
                best_coefficients = coefficients1, coefficients2

        return Candidate(index=min_index, error=min_error, time=xdata[min_index],
                         coefficients1=best_coefficients[0], coefficients2=best_coefficients[1])


def constant(xdata: np.ndarray) -> np.ndarray:
    """ Constant function

    :param xdata: Input data
    :return: All ones
    """
    return np.ones(len(xdata))


def linear(xdata: np.ndarray) -> np.ndarray:
    """ Linear function

    :param xdata: Input data.
    :return: Same as input data
    """
    return xdata


def square(xdata: np.ndarray) -> np.ndarray:
    """ Quadratic function

    :param xdata: Input data.
    :return: Square of the input data.
    """
    return xdata**2


if __name__ == "__main__":
    np.random.seed(0)
    C = ChangePointDetector([constant, linear], np.linspace(0, 1, 100),
                            np.abs(np.linspace(-1, 1, 100))+np.random.randn(100)*0.1)
    print(C.find_candidate(C.xdata, C.ydata))
