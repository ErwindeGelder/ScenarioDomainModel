""" Class for reading and adding extra information to ERP2017 data, including various activities.

Creation data: 2019 08 23
Author(s): Erwin de Gelder

Modifications:
"""

from typing import List, Tuple, NamedTuple
import pandas as pd
import numpy as np
from options import Options


LineData = NamedTuple("LineData", [("distance", pd.Series), ("difference", pd.Series)])
TargetLines = NamedTuple("LineData", [("left", pd.Series), ("right", pd.Series),
                                      ("left_up", pd.Series), ("left_down", pd.Series),
                                      ("right_up", pd.Series), ("right_down", pd.Series)])
FromGoal = NamedTuple("FromGoal", [("from_y", float), ("goal_y", float)])


class ActivityDetectorParameters(Options):
    """ Parameters that are used by the ActivityDetector. """
    # Fields of the DataFrame.
    host_long_vel = 'Host_vx'
    y_left_line = 'Host_line_left_c0'
    y_right_line = 'Host_line_right_c0'
    y_left_line_lin = 'Host_line_left_c1'
    y_left_line_sqr = 'Host_line_left_c2'
    y_left_line_cub = 'Host_line_left_c3'
    y_right_line_lin = 'Host_line_right_c1'
    y_right_line_sqr = 'Host_line_right_c2'
    y_right_line_cub = 'Host_line_right_c3'
    left_conf = 'Host_line_left_quality'
    right_conf = 'Host_line_right_quality'
    x_target = 'dx'
    y_target = 'dy'
    v_target = 'vx'
    a_target = 'ax'

    # Parameters that alter the activity detection.
    time_speed_difference = 1  # [s]
    min_speed_difference = 0.5 / 3.6  # [m/s]
    min_activity_speed = 0.25 / 3.6  # [m/s]
    diffspeed_start_act = 0.25 / 3.6  # [m/s]
    min_speed_inc = 4 / 3.6  # [m/s]
    min_cruising_time = 4  # [s]
    max_time_activity = 300  # [s] This speeds up the computation hugely
    max_time_host_lane_change = 10  # [s]
    min_line_quality = 3
    lane_change_threshold = 1  # [m]
    lane_change_magic_time = 1  # [s] No idea why this is one or why it is needed (i.e., why not 0?)
    lane_conf_threshold = 0.25
    max_time_lat_target = 6  # [s]
    factor_goal_y_target = 0.25  # ???


class ActivityDetector:
    """ Class for adding extra information to a dataframe.

    Attributes:
        data: A pandas dataframe containing all the data.
        parms: Different kinds of parameters, see ActivityDetectorParameters.
        frequency: The sample frequency of the data.
    """
    def __init__(self, data: pd.DataFrame, parameters: ActivityDetectorParameters = None):
        self.data = data
        self.parms = ActivityDetectorParameters() if parameters is None else parameters

        # Converting the frequency to an int improves speed by ~25 %.
        self.frequency = np.round(1e9 / np.mean(np.diff(self.get('Time')))).astype(int)

    def get(self, signal: str, index: float = None):
        """ Get certain data by its name.

        :param signal: The name of the signal.
        :param index: In case of a single value, the index of the value is
                      passed.
        :return: The numpy array of the data.
        """
        if index is None:
            return self.data[signal]
        return self.data.at[index, signal]

    def set(self, signal: str, data: np.ndarray) -> None:
        """ Set data.

        :param signal: The name of the signal.
        :param data: The data that is set.
        """
        self.data[signal] = data

    def get_t(self, target_index: int, signal: str, index: float = None):
        """ Get target data.

        :param target_index: The index of the target (from 0 till 7).
        :param signal: The name of the signal.
        :param index: In case of a single value, the index of the value is
                      passed.
        :return: The numpy array of the data.
        """
        signal = 'Target_{:d}_{:s}'.format(target_index, signal)
        return self.get(signal, index)

    def set_t(self, target_index: int, signal: str, data: np.ndarray) -> None:
        """ Set target data.

        :param target_index: The index of the target (from 0 till 7).
        :param signal: The name of the signal.
        :param data: The data that is set.
        """
        signal = 'Target_{:d}_{:s}'.format(target_index, signal)
        self.data[signal] = data

    def get_all_data(self) -> pd.DataFrame:
        """ Return the dataframe. """
        return self.data

    def long_activities_host(self) -> List[Tuple[float, str]]:
        """ Compute the longitudinal activities of the host vehicle.

        The activities accelerating, decelerating, and cruising of the host
        vehicle are detected. However, the events preceding the activities are
        returned. Each event is a tuple of the time and the name of the
        following activity. The name can be 'a' (accelerating),
        'd' (decelerating), and 'c' (cruising). The returned time corresponds
        to an index of the dataframe.

        :return: A list of events, where each event is a tuple of its time and
                 the name of the following activity.
        """
        # Compute speed increase in next second.
        shift = np.round(self.parms.time_speed_difference * self.frequency).astype(np.int)
        shifted = self.data[self.parms.host_long_vel].shift(-shift)
        filtered = shifted.rolling(shift).min()
        self.set('speed_inc', shifted - filtered)
        self.set('speed_inc_start', self.get('speed_inc').copy())
        self.data.loc[self.data[self.parms.host_long_vel] != filtered, 'speed_inc_start'] = 0

        # Compute speed decrease in next second.
        shifted = self.data[self.parms.host_long_vel].shift(-shift)
        filtered = shifted.rolling(shift).max()
        self.set('speed_dec', shifted - filtered)
        self.set('speed_dec_start', self.get('speed_dec').copy())
        self.data.loc[self.data[self.parms.host_long_vel] != filtered, 'speed_dec_start'] = 0

        all_events = [(self.data.index[0], 'c')]
        event = 'c'
        cruise_switch = 0
        speed_inc = self.get("speed_inc")
        speed_dec = -self.get("speed_dec")
        for row in self.data.itertuples():
            cruise_switch -= 1

            # Potential acceleration signal when in minimum wrt next second, accelerating and not
            # standing still.
            if event != 'a' and row.speed_inc_start >= self.parms.min_speed_difference and \
                    getattr(row, self.parms.host_long_vel) >= self.parms.min_activity_speed:
                i, is_event = self.end_long_activity(row.Index, speed_inc)
                if is_event:
                    event = 'a'
                    all_events.append((row.Index, event))
                    cruise_switch = (i - row.Index) * self.frequency
            elif event != 'd' and row.speed_dec_start <= -self.parms.min_speed_difference:
                i, is_event = self.end_long_activity(row.Index, speed_dec)
                if is_event:
                    event = 'd'
                    all_events.append((row.Index, event))
                    cruise_switch = (i - row.Index)*self.frequency
            elif event != 'c' and cruise_switch <= 0:
                event = 'c'
                all_events.append((row.Index, event))

        # Remove small cruise activities.
        events = []
        i = 0
        while i < len(all_events):
            if i == 0 or i == len(all_events)-1 or all_events[i][1] in ['a', 'd'] or \
                    all_events[i+1][0]-all_events[i][0] >= self.parms.min_cruising_time:
                events.append(all_events[i])
                i += 1
            elif all_events[i-1][1] == all_events[i+1][1]:
                i += 2
            else:
                xvel = self.data[self.parms.host_long_vel].loc[all_events[i][0]:all_events[i+1][0]]
                if all_events[i+1][1] == 'a':
                    events.append((xvel[::-1].idxmin(), 'a'))
                else:
                    events.append((xvel[::-1].idxmax(), 'd'))
                i += 2
        return events

    def end_long_activity(self, i: float, speed_difference: pd.Series):
        """ Find first index where speed difference is less than significant.

        :param i: Index of the start of the potential longitudinal activity.
        :param speed_difference: The speed difference (up for accelerating,
                                 down for decelerating).
        :return: A tuple (end_time, is_event). If no activity, is_event is
                 False.
        """
        end_i = np.min((i + self.parms.max_time_activity, self.data.index[-1]))
        # If someone can optimize next line... That line is responsible for 75%
        # of the executing time for longitudinal activities...
        end_i = next((j for j, value in speed_difference[i:end_i].iteritems() if
                      value < self.parms.diffspeed_start_act),
                     self.data.index[-1])

        if np.sum(speed_difference.loc[i:end_i]) / self.frequency < self.parms.min_speed_inc:
            return 0, False
        return end_i, True

    def lat_activities_host(self) -> List[Tuple[float, str]]:
        """ Compute the lateral activities of the host vehicle.

        The following events are possible:
        - fl: follow lane
        - l: A left lane change
        - r: A right lane change.
        A list of events is returned. Each event is a tuple with the time of the
        event and the activity that starts at the time of the event.

        :return: A list of events, where each event is a tuple of its time and
                 the name of the following activity.
        """
        self.set('line_center_y', (self.get(self.parms.y_left_line) +
                                   self.get(self.parms.y_right_line)) / 2)
        self.set('line_left_y_conf', self.get(self.parms.y_left_line))
        self.data.loc[self.get(self.parms.left_conf) < self.parms.min_line_quality,
                      'line_left_y_conf'] = np.nan
        self.set('line_right_y_conf', self.get(self.parms.y_right_line))
        self.data.loc[self.get(self.parms.right_conf) < self.parms.min_line_quality,
                      'line_right_y_conf'] = np.nan
        self.set('line_center_y_conf', self.get('line_center_y'))
        self.data.loc[(self.get(self.parms.left_conf) < self.parms.min_line_quality) |
                      (self.get(self.parms.right_conf) < self.parms.min_line_quality),
                      'line_center_y_conf'] = np.nan
        self.set('line_left_y_conf_down', self.get('line_left_y_conf') -
                 self.data['line_left_y_conf'].rolling(window=self.frequency).max())
        self.set('line_right_y_conf_up', self.get('line_right_y_conf') -
                 self.data['line_right_y_conf'].rolling(window=self.frequency).min())

        events = [(self.data.index[0], 'fl')]
        event = 'fl'

        previous_y = (self.get(self.parms.y_left_line, self.data.index[0]),
                      self.get(self.parms.y_right_line, self.data.index[0]))
        left_y = self.get(self.parms.y_left_line)
        left_dy = self.get('line_left_y_conf_down')
        right_y = self.get(self.parms.y_right_line)
        right_dy = self.get('line_right_y_conf_up')
        for row in self.data.itertuples():
            # Left lane change (out of ego lane).
            if event != 'l' and self.potential_left(getattr(row, self.parms.y_left_line),
                                                    previous_y[0], row.line_left_y_conf_down):
                begin_i, lane_change = self.start_lane_change(row.Index, events[-1], left_y,
                                                              left_dy)
                if lane_change:
                    event = 'l'
                    events.append((begin_i, event))

            # Right lane change (out of ego lane)
            elif event != 'r' and self.potential_right(getattr(row, self.parms.y_right_line),
                                                       previous_y[1],
                                                       row.line_right_y_conf_up):
                begin_i, lane_change = self.start_lane_change(row.Index, events[-1], -right_y,
                                                              -right_dy)
                if lane_change:
                    event = 'r'
                    events.append((begin_i, event))

            # Follow-Lane
            elif event != 'fl' and row.line_center_y_conf != 0 and \
                    row.line_right_y_conf_up < self.parms.lane_conf_threshold and \
                    row.line_left_y_conf_down > -self.parms.lane_conf_threshold:
                event = 'fl'
                events.append((row.Index, event))

            # Update the previous positions
            previous_y = (getattr(row, self.parms.y_left_line),
                          getattr(row, self.parms.y_right_line))
        return events

    def potential_left(self, current_y: float, previous_y: float,
                       lateral_difference: float) -> bool:
        """ Determine whether there is a potential left lane change of the host.

        If any of the following is true, a False is returned.
        - The current distance toward the left lane line is larger than or
          equal to 0.
        - The previous distance toward the left lane line is smaller than 0.
        - The absolute difference between the current and previous distance
          should be smaller than "lane_change_threshold".
        - The lateral_difference is NaN.
        - The lateral difference is larger than minus "lane_conf_threshold"

        :param current_y: The current distance toward the left lane line.
        :param previous_y: The previous distance toward the left lane line.
        :param lateral_difference: The difference in lateral distance to left
                                   lane compared to "time_speed_difference" ago.
        :return: A boolean whether there is a potential lane change.
        """
        if current_y >= 0:
            return False
        if previous_y < 0:
            return False
        if abs(current_y - previous_y) >= self.parms.lane_change_threshold:
            return False
        if np.isnan(lateral_difference):
            return False
        if lateral_difference >= -self.parms.lane_conf_threshold:
            return False
        return True

    def potential_right(self, current_y: float, previous_y: float,
                        lateral_difference: float) -> bool:
        """ Determine whether there is a potential right lane change of host.

        If any of the following is true, a False is returned.
        - The current distance toward the right lane line is larger than or
          equal to 0.
        - The previous distance toward the left lane line is smaller than 0.
        - The absolute difference between the current and previous distance
          should be smaller than "lane_change_threshold".
        - The lateral difference is NaN.
        - The lateral difference is larger than minus "lane_conf_threshold"

        :param current_y: The current distance toward the right lane line.
        :param previous_y: The previous distance toward the right lane line.
        :param lateral_difference: The difference in lateral distance to right
                                   lane compared to "time_speed_difference" ago.
        :return: A boolean whether there is a potential lane change.
        """
        return self.potential_left(-current_y, -previous_y, -lateral_difference)

    def start_lane_change(self, i: float, event: Tuple[float, str], lateral_distance: pd.Series,
                          lateral_difference: pd.Series) -> Tuple[float, bool]:
        """ Compute the start of a potential lane change.

        If there is no lane change found, (0, False) will be returned.
        If a lane change is found, (begin_time, True) will be returned.

        :param i: The index at which the potential lane change is found.
        :param event: The last event (needed for the last event time).
        :param lateral_distance: The lateral distance toward the left side in
                                 case of a left lane change or right side in
                                 case of a right lane change.
        :param lateral_difference: The difference of the lateral_distance
                                   compared to "time_speed_difference" ago.
        :return: starting index and whether there is a lane change.
        """
        begin_i = np.max((i - self.parms.max_time_host_lane_change, event[0]))
        begin_i = next((j for j, value in zip(lateral_difference[begin_i:i].index[::-1],
                                              lateral_difference[begin_i:i].values[::-1]) if
                        value > -self.parms.lane_conf_threshold), None)
        if begin_i is None:
            return 0, False

        begin_i = lateral_distance[np.max((begin_i - self.parms.lane_change_magic_time,
                                           self.data.index[0])):begin_i].idxmax()
        next_y = lateral_distance.iat[self.data.index.get_loc(begin_i) + 1]
        if abs(lateral_distance[begin_i] - next_y) > self.parms.lane_change_threshold:
            return 0, False

        return begin_i, True

    def long_activities_target_i(self, i: int) -> List[Tuple[float, str]]:
        """ Compute the longitudinal activities of the i-th target vehicle.

        The activities cruising ("c"), decelerating ("d"), and accelerating
        ("a") are detected. To do this, a new ActivityDetector is constructed
        with the target vehicle as "Host" vehicle.
        Automatically, the same thresholds are used as for the host vehicle.

        :param i: Index of the target.
        :return: A list of events, where each event is a tuple of its time and
                 the name of the following activity.
        """
        parameters = self.parms
        parameters.host_long_vel = "Target_{:d}_{:s}".format(i, self.parms.v_target)
        target_df = self.data[[parameters.host_long_vel, 'Time']].copy()
        activity_detector = ActivityDetector(target_df, parameters=parameters)
        return activity_detector.long_activities_host()

    def line_info_target_i(self, i: int):
        """ Compute the lateral activities of the i-th target vehicle.

        :param i: Index of the target.
        """
        line_left = (self.get(self.parms.y_left_line) +
                     self.get(self.parms.y_left_line_lin) *
                     self.get_t(i, self.parms.x_target) +
                     self.get(self.parms.y_left_line_sqr) *
                     np.power(self.get_t(i, self.parms.x_target), 2) +
                     self.get(self.parms.y_left_line_cub) *
                     np.power(self.get_t(i, self.parms.x_target), 3) -
                     self.get_t(i, self.parms.y_target))
        line_right = (self.get(self.parms.y_right_line) +
                      self.get(self.parms.y_right_line_lin) *
                      self.get_t(i, self.parms.x_target) +
                      self.get(self.parms.y_right_line_sqr) *
                      np.power(self.get_t(i, self.parms.x_target), 2) +
                      self.get(self.parms.y_right_line_cub) *
                      np.power(self.get_t(i, self.parms.x_target), 3) -
                      self.get_t(i, self.parms.y_target))
        line_left[self.get(self.parms.left_conf) < self.parms.min_line_quality] = np.nan
        line_right[self.get(self.parms.right_conf) < self.parms.min_line_quality] = np.nan
        line_center_y = (line_left + line_right) / 2
        self.set_t(i, 'line_left', line_left)
        self.set_t(i, 'line_right', line_right)
        self.set_t(i, 'line_center', line_center_y)
        self.set_t(i, 'line_left_down',
                   (self.get_t(i, 'line_left') -
                    self.get_t(i, 'line_left').rolling(window=self.frequency).max()))
        self.set_t(i, 'line_left_up',
                   (self.get_t(i, 'line_left') -
                    self.get_t(i, 'line_left').rolling(window=self.frequency).min()))
        self.set_t(i, 'line_right_down',
                   (self.get_t(i, 'line_right') -
                    self.get_t(i, 'line_right').rolling(window=self.frequency).max()))
        self.set_t(i, 'line_right_up',
                   (self.get_t(i, 'line_right') -
                    self.get_t(i, 'line_right').rolling(window=self.frequency).min()))

    def lat_activities_target_i(self, i: int) -> List[Tuple[float, str]]:
        """ Compute the lateral activities of the i-th target vehicle.

        The following events are possible:
        - fl: follow lane
        - lo: A left lane change out of the lane of the ego vehicle.
        - li: A left lane change into the lane of the ego vehicle.
        - ro: A right lane change out of the lane of the ego vehicle.
        - ri: A right lane change into the lane of the ego vehicle.
        A list of events is returned. Each event is a tuple with the time of the
        event and the activity that starts at the time of the event.

        :param i: Index of the target.
        :return: A list of events, where each event is a tuple of its time and
                 the name of the following activity.
        """
        self.line_info_target_i(i)

        event = 'fl'
        prev_index = self.data.index[0]
        events = [(prev_index, event)]
        follow_lane_switch = 0

        # We are going to grab all line data. By doing this once, we speed up the calculation at a
        # small cost of memory usage.
        lines = TargetLines(left=self.get_t(i, 'line_left'),
                            right=self.get_t(i, 'line_right'),
                            left_down=self.get_t(i, 'line_left_down'),
                            right_down=self.get_t(i, 'line_right_down'),
                            left_up=self.get_t(i, 'line_left_up'),
                            right_up=self.get_t(i, 'line_right_up'))
        for row in self.data.iloc[1:].itertuples():
            follow_lane_switch -= 1
            valid = (not np.isnan(row.line_center_y) and
                     get_from_row(row, 'line_right', i) < get_from_row(row, 'line_left', i))

            # Left lane change out of ego lane: cross left_y down.
            if valid and event != 'lo' and event != 'li' and \
                    self.potential_left(get_from_row(row, 'line_left', i), lines.left[prev_index],
                                        get_from_row(row, 'line_left_down', i)):
                lineinfo = LineData(distance=lines.left, difference=lines.left_down)
                begin_j, end_j, lane_change = self.start_end_target(i, row, lineinfo, events[-1])
                if lane_change:
                    event = 'lo'
                    events.append((begin_j, event))
                    follow_lane_switch = (end_j - row.Index)*self.frequency

            # Left lane change into ego lane: cross right_y down.
            elif valid and event != 'li' and event != 'lo' and \
                    self.potential_left(get_from_row(row, 'line_right', i), lines.right[prev_index],
                                        get_from_row(row, 'line_right_down', i)):
                lineinfo = LineData(distance=lines.right, difference=lines.right_down)
                begin_j, end_j, lane_change = self.start_end_target(i, row, lineinfo, events[-1])
                if lane_change:
                    event = 'li'
                    events.append((begin_j, event))
                    follow_lane_switch = (end_j - row.Index)*self.frequency

            # Right lane change into ego lane: cross left_y up.
            elif valid and event != 'ri' and event != 'ro' and \
                    self.potential_right(get_from_row(row, 'line_left', i), lines.left[prev_index],
                                         get_from_row(row, 'line_left_up', i)):
                lineinfo = LineData(distance=-lines.left, difference=lines.left_up)
                begin_j, end_j, lane_change = self.start_end_target(i, row, lineinfo, events[-1])
                if lane_change:
                    event = 'ri'
                    events.append((begin_j, event))
                    follow_lane_switch = (end_j - row.Index)*self.frequency

            # Right lane change out of ego lane: cross right_y up.
            elif valid and event != 'ro' and event != 'ri' and \
                    self.potential_right(get_from_row(row, 'line_right', i),
                                         lines.right[prev_index],
                                         get_from_row(row, 'line_right_up', i)):
                lineinfo = LineData(distance=-lines.right, difference=lines.right_up)
                begin_j, end_j, lane_change = self.start_end_target(i, row, lineinfo, events[-1])
                if lane_change:
                    event = 'ro'
                    events.append((begin_j, event))
                    follow_lane_switch = (end_j - row.Index)*self.frequency

            elif event != 'fl' and follow_lane_switch <= 0:
                event = 'fl'
                events.append((row.Index, event))
            prev_index = row.Index
        return events

    def start_lc_target(self, j: float, fromgoal: FromGoal, lineinfo: LineData,
                        event: Tuple[float, str]) -> Tuple[float, bool]:
        """ Compute the starting index of a potential lane change.

        :param j: The index at which the potential lane change is found.
        :param fromgoal: Containing the y-positions of the start and goal of the
                         target.
        :param lineinfo: The distance toward the crossing line and its
                         derivative.
        :param event: The last event.
        :return: The starting index and a boolean whether there is a potential
                 lane change.
        """
        begin_j = np.max((j - self.parms.max_time_lat_target, event[0]))
        start_found = False
        for begin_j, distance in zip(lineinfo.distance[begin_j:j].index[::-1],
                                     lineinfo.distance[begin_j:j][::-1]):
            if begin_j != j and distance < fromgoal.goal_y/2:
                # Why is goal_y divided by 2?
                return 0, False
            if distance > fromgoal.from_y:
                if begin_j == self.data.index[0]:
                    start_found = True
                    break
                prev_begin_j = self.data.index[self.data.index.get_loc(begin_j) - 1]
                if abs(distance - lineinfo.distance[prev_begin_j]) < \
                        self.parms.lane_change_threshold:
                    start_found = True
                break
        if start_found:
            if np.any(lineinfo.difference[begin_j:j] == 0):
                begin_j = \
                    lineinfo.difference[begin_j:j][lineinfo.difference[begin_j:j] == 0].index[-1]
            return begin_j, True
        return 0, False

    def end_lc_target(self, j, fromgoal, line_y):
        """ Compute the end index of a potential lane change.

        :param j: The index at which the potential lane change is found.
        :param fromgoal: Containing the y-positions of the start and goal of the
                         target.
        :param line_y: The distance toward the crossing.
        :return: The end index and a boolean whether there is a lane change.
        """
        end_j = np.min((j + self.parms.max_time_lat_target, self.data.index[-1]))
        for end_j, distance in zip(line_y[j:end_j].index, line_y[j:end_j]):
            if distance > 0:
                return 0, False
            if distance < fromgoal.goal_y:
                prev_end_j = self.data.index[self.data.index.get_loc(end_j) - 1]
                if abs(distance - line_y[prev_end_j]) < self.parms.lane_change_threshold:
                    return end_j, True
                break
        return 0, False

    def start_end_target(self, i: int, row, lineinfo: LineData,
                         event: Tuple[float, str]) -> Tuple[float, float, bool]:
        """ Compute the start and end of a potential lane change of a target.

        If the potential lane change is not a lane change, (0, 0, False) will be
        returned.
        There is no type hinting for the argument "row", because it should be
        returned by pandas itertuples(), which returns a new "class". However,
        it should contain the fields "line_left" and "line_right" of target i
        and it should contain the "Index" of the row.

        :param i: Index of target.
        :param row: A row of the dataframe.
        :param lineinfo: The distance toward the crossing line and its
                         derivative.
        :param event: The last event.
        :return: The starting index, the end index, and a boolean whether there
                 is a lane change.
        """
        fromgoal = self.goal_left_lane_change(get_from_row(row, 'line_left', i),
                                              get_from_row(row, 'line_right', i))
        begin_j, lane_change = self.start_lc_target(row.Index, fromgoal, lineinfo, event)
        if lane_change:
            end_j, lane_change = self.end_lc_target(row.Index, fromgoal, lineinfo.distance)
            return begin_j, end_j, lane_change
        return 0, 0, False

    def goal_left_lane_change(self, left_y: float, right_y: float) -> FromGoal:
        """ Compute the goal and the starting position of a left lane change.

        :param left_y: Distance toward left line.
        :param right_y: Distance toward right line.
        :return: The lateral position from which the lane change start and where
                 it ends.
        """
        from_y = self.parms.factor_goal_y_target * (left_y - right_y)
        goal_y = -from_y
        return FromGoal(from_y=from_y, goal_y=goal_y)


def get_from_row(row, signal, target_index: int = None) -> float:
    """ Get data entry from a row of data.

    :param row: Row of a data. Should be a named tuple, obtained via
                itertuples().
    :param signal: The name of the signal.
    :param target_index: The index of the target, if the signal corresponds
                         to a target.
    :return: The value of the entry.
    """
    if target_index is not None:
        return getattr(row, 'Target_{:d}_{:s}'.format(target_index, signal))
    return getattr(row, signal)
