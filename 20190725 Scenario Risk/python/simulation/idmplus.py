""" Intelligent Driver Model plus (IDM+) from Schakel et al. (2010).

Creation date: 2020 05 28
Author(s): Erwin de Gelder

Modifications:
"""

import numpy as np
from .idm import IDM


class IDMPlus(IDM):
    """ Class for simulation of the Intelligent Driver Model plus (IDM+). """
    def __init__(self):
        IDM.__init__(self)

    def _acceleration(self, xlead: float, vlead: float) -> float:
        """ Compute the acceleration.

        This is where IDM+ differs from IDM. Instead of superimposing the free
        flow term and the non-free flow term, the IDM+ takes the minimum
        between the two.

        :param xlead: Position of leading vehicle.
        :param vlead: Speed of leading vehicle.
        :return: The acceleration.
        """
        return self.parms.a_acc * np.min((1 - self._freeflowpart(),
                                          1 - self._nonfreeflowpart(xlead, vlead)))
