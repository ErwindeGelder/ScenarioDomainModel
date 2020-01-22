""" Glueing targets together if they get lost in a blind spot.

Creation date: 2019 12 24
Author(s): Erwin de Gelder

Modifications:
2019 12 30 Change the way options are specified.
2020 01 12 Fixing issue with age of target when merging two targets.
2020 01 21 Use the host speed when target is missing instead of assuming constant host speed.
2020 01 22 Use average speed for target and candidate instead of only the target speed.
"""

from typing import List, NamedTuple
import numpy as np
import pandas as pd
from data_handler import DataHandler
from options import Options


class _TargetGluerOptions(Options):
    """ Different configuration options for the target gluer. """
    tsec_v_avg: float = 1  #
    t_min_available: float = 0.5  # [s]
    x_min_visible: float = -10  # min x distance at which the target is (still) visible
    x_max_visible: float = 25  # max x distance at which the target is (still) visible
    minimum_speed: float = 1

    t_max_gone: float = 15  # [s]
    x_deviation: float = 5  # [m]
    v_deviation: float = 5  # [m/s]

    fieldname_host_vx: str = "Host_vx"
    fieldname_target_vx: str = "vx"
    fieldname_target_dx: str = "dx"
    fieldname_target_dy: str = "dy"
    fieldname_target_id: str = "id"
    fieldname_target_age: str = "age"

    verbose: bool = False


def merge_targets(datahandler: DataHandler, **kwargs) -> None:
    """ Merge targets if they get lost in the sensors' blind spot.

    The list of the targets will be adjusted. Hence, nothing is returned.
    With the **kwargs, the way the targets are merged can be specified. The
    following options are available:
    - tsec_v_avg: float, default 1
        The time in seconds over which the speed is averaged at the end and
        start of a target.
    - t_min_available: float, default 0.5
        The minimum time for a target to be considered for merging.
    - x_min_visible: float, default -10
        All targets with x < x_min_visible are NOT in the blind spot.
    - x_max_visible: float, default 20
        All targets with x > x_max_visible are NOT in the blind spot.
    - minimum_speed: float, default 1
        Minimum relative speed for a target vehicle when entering the blind spot
        to be considered for merging.
    - t_max_gone: float, default 15
        The maximum time a target might be out of view.
    - x_deviation: float, default 5
        The maximum deviation in x-position for two targets to be considered for
        merging.
    - v_deviation: float, default 5
        The maximum deviation in speed for two targets to be considered for
        merging.
    - fieldname_host_vx: str, default "Host_vx"
        The fieldname for the speed of the host vehicle.
    - fieldname_target_vx: str, default "vx"
        The fieldname for the speed of the targets.
    - fieldname_target_dx: str, default "dx"
        The fieldname for the relative longitudinal position of the targets.
    - fieldname_target_dy: str, default "dy"
        The fieldname for the relative lateral position of the targets.
    - fieldname_target_id: str, default "id"
        The fieldname for the ID of the targets.
    - fieldname_target_age: str, default "age"
        The fieldname for the age of the targets.
    - verbose: bool, default False
        If True, information is printed (might be useful for inspection).

    :param datahandler: handler of the data.
    :param kwargs: Options for the target merger, see description above.
    """
    host = datahandler.data
    targets = datahandler.targets
    timestep = 1/datahandler.frequency
    options = _TargetGluerOptions(**kwargs)
    _TargetGluer(host, targets, timestep, options)


class _TargetInfo(NamedTuple):
    """ Information of a target that might be in the blind spot. """
    is_blind_spot: bool
    is_front: bool = False
    last_x: float = 0
    last_y: float = 0
    last_t: float = 0
    vx_host: float = 0
    vx_target: float = 0


class _CandidateInfo(NamedTuple):
    """ Information of a candidate that matches with a target."""
    is_candidate: bool
    stop_searching: bool = False
    x_offset: float = 0


class _TargetGluer:
    """ Glue targets together if they get lost in a blind spot. """
    def __init__(self, host: pd.DataFrame, targets: List[pd.DataFrame], timestep: float,
                 options: _TargetGluerOptions):
        self.host = host
        self.targets = targets
        self.timestep = timestep
        self.options = options

        # Go through all targets and glue with other targets if targets are likely to be the same.
        for i, target in enumerate(self.targets):
            while self.search_and_merge(i, target):
                pass

    def search_and_merge(self, i_target: int, target: pd.DataFrame) -> bool:
        """ Search for another target and if found, merge the targets.

        :param i_target: Number of the target.
        :param target: Dataframe of the corresponding target.
        :return: Whether another target is found or not.
        """
        info = self.target_in_blind_spot(target)
        if not info.is_blind_spot:
            return False

        # Go through all remaining targets and see if we can match it.
        minimum_x_offset = np.inf
        i_best_candidate = 0
        for i_candidate, candidate in enumerate(self.targets[i_target+1:], start=i_target+1):
            candidate_info = self.target_match_candidate(candidate, info, minimum_x_offset)
            if candidate_info.stop_searching:
                break
            if not candidate_info.is_candidate:
                continue

            minimum_x_offset = candidate_info.x_offset
            i_best_candidate = i_candidate

        if i_best_candidate == 0:
            return False

        # Merge the targets and remove the candidate from the list.
        candidate = self.targets[i_best_candidate]
        if self.options.verbose:
            print("Merge target {:.0f} with target {:.0f}.".
                  format(candidate[self.options.fieldname_target_id].values[0],
                         target[self.options.fieldname_target_id].values[0]))
        candidate[self.options.fieldname_target_id] = \
            target[self.options.fieldname_target_id].values[0]
        candidate[self.options.fieldname_target_age] += \
            (candidate.index[0] - target.index[-1])*1e9 + \
            target[self.options.fieldname_target_age].values[-1]
        new_index = np.concatenate((target.index,
                                    np.round(np.arange(info.last_t, candidate.index[0],
                                                       self.timestep)[1:], 2),
                                    candidate.index))
        new_index = np.unique(new_index)
        target = pd.concat((target, candidate))
        target = target.reindex(new_index).interpolate(method='linear')
        self.targets.pop(i_best_candidate)
        self.targets[i_target] = target
        return True

    def target_in_blind_spot(self, target: pd.DataFrame) -> _TargetInfo:
        """ Determine if a target disappears in the blind spot.

        :param target: The dataframe of the target.
        :return: Information regarding the target that could be used for
                 matching with another target.
        """
        # Target should exist long enough to be considered.
        if target.index[-1] - target.index[0] <= self.options.t_min_available:
            return _TargetInfo(False)

        # Compute the mean speed over <tsec_v_avg>.
        vx_target = np.mean(target[self.options.fieldname_target_vx][target.index[-1] -
                                                                     self.options.tsec_v_avg:])
        vx_host = self.host.at[target.index[-1], self.options.fieldname_host_vx]

        # Relative speed may not be too small.
        if abs(vx_target - vx_host) <= self.options.minimum_speed:
            return _TargetInfo(False)

        # Determine if we could expect the target to go into the blind spot.
        last_x = target[self.options.fieldname_target_dx].values[-1]
        is_blind_spot = False
        is_front = False
        if 0 < last_x < self.options.x_max_visible:
            # If target is in front relative speed must be negative to enter blind spot.
            if vx_target < vx_host:
                is_blind_spot = True
                is_front = True
        elif self.options.x_min_visible < last_x < 0:
            # If target behind relative speed must be positive to enter blind spot.
            if vx_target > vx_host:
                is_blind_spot = True
                is_front = False

        if not is_blind_spot:
            return _TargetInfo(False)

        last_y = target[self.options.fieldname_target_dy].values[-1]
        return _TargetInfo(True, is_front=is_front, last_x=last_x, last_y=last_y,
                           vx_target=vx_target, vx_host=vx_host, last_t=target.index[-1])

    def target_match_candidate(self, candidate: pd.DataFrame, info: _TargetInfo,
                               max_offset: float) -> _CandidateInfo:
        """ Determine if the candidate is the same vehicle as the target.

        :param candidate: dataframe of the candidate vehicle.
        :param info: information of the target vehicle.
        :param max_offset: the maximum offset of other candidates.
        :return: information about this candidate.
        """
        # Candidate should become visible not too long after target disappeared.
        t_gone = candidate.index[0] - info.last_t
        if t_gone > self.options.t_max_gone:
            return _CandidateInfo(False, stop_searching=True)

        # Candidate should become visible after the target disappeared.
        # Candidate should be visible for long enough.
        if t_gone <= 0 or candidate.index[-1] - candidate.index[0] <= self.options.t_min_available:
            return _CandidateInfo(False)

        # Candidate should be in front if target was in back and vice versa.
        # x-position should be in the range [x_min_visible, x_max_visible].
        x_candidate = candidate[self.options.fieldname_target_dx].values[0]
        if (info.is_front and x_candidate > 0) or (not info.is_front and x_candidate < 0) or \
                (not self.options.x_min_visible < x_candidate < self.options.x_max_visible):
            return _CandidateInfo(False)

        # x-position should not be too much off.
        # Sign of y-position should be similar.
        speed_start = candidate[self.options.fieldname_target_vx][:candidate.index[0] +
                                                                  self.options.tsec_v_avg]
        speed_start = np.mean(speed_start[speed_start > 0])
        x_absoffset = abs(info.last_x + t_gone*(info.vx_target+speed_start)/2 - x_candidate -
                          np.trapz(self.host.loc[info.last_t:candidate.index[0],
                                                 self.options.fieldname_host_vx])*self.timestep)
        if x_absoffset > self.options.x_deviation or x_absoffset > max_offset or \
                np.sign(info.last_y) != np.sign(candidate[self.options.fieldname_target_dy].iat[0]):
            return _CandidateInfo(False)

        # Speed should not be too different.
        if abs(speed_start - info.vx_target) > self.options.v_deviation:
            return _CandidateInfo(False)

        return _CandidateInfo(True, x_offset=x_absoffset)
