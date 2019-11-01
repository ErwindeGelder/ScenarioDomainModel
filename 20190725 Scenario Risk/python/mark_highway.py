""" Mark the parts where the ego vehicle drives on the highway.

Creation date: 2019 11 01
Author(s): Erwin de Gelder

Modifications:
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import utm
from options import Options


class HighwayMarkerOptions(Options):
    """ Options for the HighwayMarker. """
    minimal_distance: float = 15  # [m]
    change_highway_dist = 20  # [m]
    max_angle_diff = np.pi / 3

    # Define the coordinates at which the highway is entered and exited.
    # Coordinates are in [lat, lon, angle]
    enter: List[Tuple[float, float, float]] = [(52.1159, 5.30590, 260),
                                               (52.1357, 5.39948, 80),
                                               (52.1948, 5.40042, 130),
                                               (52.1449, 5.41163, 60),
                                               (52.1714, 5.43751, 280),
                                               (52.1348, 5.39654, 230),
                                               (52.1076, 5.26641, 60),
                                               (52.2068, 5.36748, 90),
                                               (52.1687, 5.56859, 180),
                                               (52.0915, 5.18328, 50)]
    exit: List[Tuple[float, float, float]] = [(52.1071, 5.26014, 310),
                                              (52.1930, 5.40566, 310),
                                              (52.1163, 5.31069, 260),
                                              (52.1440, 5.41078, 70),
                                              (52.1582, 5.42090, 80),
                                              (52.1147, 5.30265, 100),
                                              (52.2088, 5.36826, 300),
                                              (52.1663, 5.56222, 120),
                                              (52.0934, 5.18358, 320)]


class HighwayMarkerParms(Options):
    """ Parameters for the HighwayMarker. """
    enter_utm: List[Tuple[float, float, float]] = []
    exit_utm: List[Tuple[float, float, float]] = []


class HighwayMarker:
    """ Adding various information to a dataframe.

    This class can be used to add whether the ego vehicle is driving on the
    highway. To compute this, the latlon data is converted to UTM coordinates.

    Attributes:
        data(pd.DataFrame): The dataframe that is used.
        options(HighwayMarkerOptions): Options the control how data is
                                       processed.
        parms(HighwayMarkerParms): Parameters of the HighwayMarker.
    """
    def __init__(self, data: pd.DataFrame, options: HighwayMarkerOptions = None):
        self.data = data
        self.options = HighwayMarkerOptions() if options is None else options
        self.parms = HighwayMarkerParms()

        # Convert coordinates of entering and exiting highway to UTM.
        self.parms.enter_utm = np.zeros((len(self.options.enter), 3))
        self.parms.exit_utm = np.zeros((len(self.options.exit), 3))
        for i, enter_latlon in enumerate(self.options.enter):
            utm_x, utm_y, _, _ = utm.from_latlon(enter_latlon[0], enter_latlon[1])
            self.parms.enter_utm[i, :2] = [utm_x, utm_y]
            self.parms.enter_utm[i, 2] = enter_latlon[2] / 180 * np.pi
        for i, exit_latlon in enumerate(self.options.exit):
            utm_x, utm_y, _, _ = utm.from_latlon(exit_latlon[0], exit_latlon[1])
            self.parms.exit_utm[i, :2] = [utm_x, utm_y]
            self.parms.exit_utm[i, 2] = exit_latlon[2] / 180 * np.pi

    def mark_highway(self) -> None:
        """ Add column with a boolean that indicates highway driving. """
        # Convert the GPS data from latlon to UTM coordinates.
        field_names = ["utm_x", "utm_y", "utm_zone", "utm_hemisphere"]
        for name in field_names:
            self.data[name] = np.zeros(len(self.data))
        utm_result = [utm.from_latlon(row.gps_lat, row.gps_lon) for
                      row in self.data[["gps_lat", "gps_lon"]].itertuples()]
        self.data[["utm_x", "utm_y", "utm_zone", "utm_hemisphere"]] = utm_result

        # Add angle of ego vehicle based on GPS.
        j = 1
        utm_x = self.data["utm_x"]
        utm_y = self.data["utm_y"]
        angle = np.zeros(len(self.data))
        for i, row in enumerate(self.data[["utm_x", "utm_y"]].itertuples()):
            if j is not None:
                j = next((j for j in range(j, len(self.data)) if
                          np.hypot(row.utm_x - utm_x.iat[j],
                                   row.utm_y - utm_y.iat[j]) > self.options.minimal_distance),
                         None)
            if j is not None:
                angle[i] = np.arctan2(utm_x.iat[j] - row.utm_x, utm_y.iat[j] - row.utm_y)
            else:
                angle[i] = angle[i - 1]
        self.data["heading"] = angle

        # Mark highway data
        highway = np.zeros(len(self.data), dtype=np.bool)
        on_highway = False
        for i, row in enumerate(self.data[["utm_x", "utm_y", "heading"]].itertuples()):
            if on_highway:  # Look for exit.
                if self.small_distance(row.utm_x, row.utm_y, row.heading, self.parms.exit_utm):
                    on_highway = False
            else:
                if self.small_distance(row.utm_x, row.utm_y, row.heading, self.parms.enter_utm):
                    on_highway = True
            highway[i] = on_highway
        self.data["is_highway"] = highway

    def small_distance(self, utm_x: float, utm_y: float, heading: float,
                       utm_points: List[Tuple[float, float, float]]) -> bool:
        """ Return whether the given point is near one of the list of points.

        :param utm_x: The UTM x coordinate of the given point.
        :param utm_y: The UTM y coordinate of the given point.
        :param heading: The heading of the given point.
        :param utm_points: The list of points which are checked.
        :return: Whether the given point is near one of the list of points.
        """
        for utm_point in utm_points:
            if np.hypot(utm_x - utm_point[0],
                        utm_y - utm_point[1]) < self.options.change_highway_dist and \
                    not (self.options.max_angle_diff <
                         np.mod(heading - utm_point[2], 2*np.pi) <
                         (2*np.pi - self.options.max_angle_diff)):
                return True
        return False
