""" Class for general functions for handling the data

Creation date: 2019 12 09
Author(s): Erwin de Gelder

Modifications:
2019 12 15 Group data by target id to create separate dataframe for each target.
2019 12 19 Provide the option to give the frequency.
2019 12 26 Save targets too.
2019 12 27 Add functionality for computed next/previous valid measurement.
2019 12 30 Add functionality for converting big target dataframe to list of targets.
"""

from typing import Any, List, Union, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


class DataHandler:
    """ Class for handling a dataframe for scenario mining purposes.

    Attributes:
        data: A pandas dataframe containing all the data OR a path to an HDF5.
        frequency: The sample frequency of the data.
        n_trackers: The number of trackers (might be 0).
        targets: List of dataframes of the targets.
    """
    def __init__(self, data: Union[pd.DataFrame, str], targets: List[pd.DataFrame] = None,
                 frequency: int = None):
        self.targets = targets
        if isinstance(data, str):
            # If a string is given, we assume that the string refers to the file with the data.
            self.read_hdf(data)
        else:
            self.data = data

        # Converting the frequency to an int improves speed by ~25 %.
        if frequency is None:
            self.frequency = np.round(1 / np.mean(np.diff(self.data.index))).astype(int)
        else:
            self.frequency = frequency

        # Determine the number of trackers.
        self.n_trackers = 0
        while self.target_signal(self.n_trackers, "id") in self.data.keys():
            self.n_trackers += 1

        # If targets are not already given, get them from the trackers.
        if self.targets is None:
            self.targets = self.create_target_dfs()

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

    def set(self, signal: str, data: np.ndarray) -> str:
        """ Set data.

        :param signal: The name of the signal.
        :param data: The data that is set.
        :return: The name of the field of the dataframe (same as signal).
        """
        self.data[signal] = data
        return signal

    @staticmethod
    def target_signal(target_index: int, signal: str):
        """ Return the name of the datafield.

        Currently, this fieldname is of the form
        "Target_<target_index>_<signal>".
        """
        return "Target_{:d}_{:s}".format(target_index, signal)

    def get_t(self, target_index: int, signal: str, index: float = None):
        """ Get target data.

        :param target_index: The index of the target (from 0 till 7).
        :param signal: The name of the signal.
        :param index: In case of a single value, the index of the value is
                      passed.
        :return: The numpy array of the data.
        """
        if index is None:
            return self.targets[target_index][signal]
        return self.targets[target_index].at[index, signal]

    def set_t(self, target_index: int, signal: str, data: Union[np.ndarray, float, str]) -> str:
        """ Set target data.

        :param target_index: The index of the target (from 0 till 7).
        :param signal: The name of the signal.
        :param data: The data that is set.
        :return: The name of the field of the dataframe.
        """
        self.targets[target_index][signal] = data
        return signal

    def get_from_row(self, row, signal, target_index: int = None) -> Any:
        """ Get data entry from a row of data.

        :param row: Row of a data. Should be a named tuple, obtained via
                    itertuples().
        :param signal: The name of the signal.
        :param target_index: The index of the target, if the signal corresponds
                             to a target.
        :return: The value of the entry.
        """
        if target_index is not None:
            return getattr(row, self.target_signal(target_index, signal))
        return getattr(row, signal)

    def get_all_data(self) -> pd.DataFrame:
        """ Return the dataframe.

        :return: The dataframe.
        """
        return self.data

    def get_target_signals(self) -> List[str]:
        """ Get all signals that a target has.

        :return: List of the keys of a target.
        """
        start = self.target_signal(0, "")
        return [key[len(start):] for key in self.data.keys() if key.startswith(start)]

    def fix_tracker_switching(self, progress_bar=False) -> int:
        """ Make sure that targets do not switch from tracker.

        :param progress_bar: Show a progress bar (default=False).
        :return: Number of switches of targets to other trackers.
        """
        signals = self.get_target_signals()
        previous_ids = np.zeros(self.n_trackers)
        switches = 0
        for row in tqdm(self.data.itertuples(), total=len(self.data), disable=not progress_bar):
            current_ids = [self.get_from_row(row, "id", j) for j in range(self.n_trackers)]
            set_to_tracker, switch = self._set_tracker(previous_ids, current_ids)
            switches += switch

            # Check if we need to replace one.
            for j, tracker in enumerate(set_to_tracker):
                if 0 <= tracker != j:
                    for signal in signals:
                        self.data.at[row.Index, self.target_signal(tracker, signal)] = \
                            self.get_from_row(row, signal, j)
                if j not in set_to_tracker:
                    for signal in signals:
                        self.data.at[row.Index, self.target_signal(j, signal)] = 0.0

            previous_ids = [self.get_t(j, "id", row.Index) for j in range(self.n_trackers)]

        return switches

    @staticmethod
    def _set_tracker(previous_ids: List, current_ids: List) -> Tuple[np.ndarray, int]:
        """ Set tracker for target.

        :param previous_ids: List of ids of previous data sample.
        :param current_ids: List of ids of current data sample.
        :return: List of trackers per target and number of switches.
        """
        # Determine tracker for targets that are also tracked in the previous data sample.
        # Also mark new targets.
        switches = 0
        set_to_tracker = -np.ones(8, dtype=np.int)
        new_targets = np.zeros(8, dtype=np.bool)
        for j, current_id in enumerate(current_ids):
            if current_id == 0:
                continue
            if current_id == previous_ids[j]:
                set_to_tracker[j] = j
            elif current_id in previous_ids:
                # Tracker has changed! This needs to be fixed.
                set_to_tracker[j] = previous_ids.index(current_id)
                switches += 1
            else:
                # New target. Mark this tracker, so we can assign a tracker later on.
                new_targets[j] = True

        # Check all new targets and assign tracker for them.
        for j, new in enumerate(new_targets):
            if new:
                # Check if we can use the same tracker.
                if j not in set_to_tracker:
                    set_to_tracker[j] = j
                    continue

                # Find a new tracker.
                for i_tracker, also_new in enumerate(new_targets):
                    if i_tracker not in set_to_tracker and not also_new:
                        set_to_tracker[j] = i_tracker
                        break

        return set_to_tracker, switches

    def create_target_dfs(self) -> List[pd.DataFrame]:
        """ Create separate dataframes for all targets.

        :return: List of dataframes where each dataframe refers to a target.
        """
        # No need to do this if we do not have any tracker.
        if self.n_trackers == 0:
            return []

        # Collect all data from the trackers.
        signals = self.get_target_signals()
        dfsubs = []
        for i in range(self.n_trackers):
            rename = {}
            for signal in signals:
                rename["Target_{:d}_{:s}".format(i, signal)] = signal
            dfsub = self.data[rename.keys()]
            dfsub = dfsub.rename(columns=rename)
            dfsubs.append(dfsub)

        # Stack all data and then group it by id.
        targets = pd.concat(dfsubs, sort=False)
        return self._big_target_df_to_list(targets, skip_first=True, sort_index=True)

    @staticmethod
    def _big_target_df_to_list(targets: pd.DataFrame, skip_first: bool = False,
                               sort_index: bool = False) -> List[pd.DataFrame]:
        """ Convert a dataframe with all target information to separate dfs.

        Each dataframe has only one value of the column "id". Therefore, each
        dataframe should refer to only one target vehicle.

        :param targets: Initial dataframe with all target information.
        :param skip_first: Whether to skip the first target.
        :param sort_index: Whether to sort the index for each target.
        :return: The list of dataframes.
        """
        targets = list(targets.groupby("id"))
        if skip_first:
            targets = targets[1:]  # Skip the first as it might refer to id=0 (no target).
        targets = [target[1] for target in targets]
        if sort_index:
            targets = [target.sort_index() for target in targets]
        return targets

    def set_diff(self, mask: pd.Series, name: str, max_valid_time: float,
                 i_target: int = None) -> None:
        """ Compute the difference between consecutive valid measurements.

        The valid input is determined by the vector `mask`. The following fields
        are created:
        - <name>_prev: The previous valid value.
        - <name>_next: The next valid value that is at least one sample ahead.
        - <name>_diff: The difference between the previous two.
        If the time between the current sample and the last (next) sample is
        more that <max_valid_time>, the last (next) sample is set to np.nan.
        By default, the host data is changed. If the data, however, concerns a
        target, its index needs to be set by <i_target>. In that case, the data
        "targets[i_target]" will be updated.

        :param mask: Indicating which values are valid.
        :param name: The name to give to the new signals.
        :param max_valid_time: The maximum time that a signal can be invalid.
        :param i_target: If data concerns a target, the index of the target.
        """
        data = self.data[name] if i_target is None else self.targets[i_target][name]
        prev_value, next_value = self._compute_prev_and_next(data, mask, max_valid_time)

        # Set the columns in the dataframe.
        if i_target is None:
            self.set("{:s}_prev".format(name), prev_value)
            self.set("{:s}_next".format(name), next_value)
            self.set("{:s}_diff".format(name), next_value-prev_value)
        else:
            self.set_t(i_target, "{:s}_prev".format(name), prev_value)
            self.set_t(i_target, "{:s}_next".format(name), next_value)
            self.set_t(i_target, "{:s}_diff".format(name), next_value-prev_value)

    def _compute_prev_and_next(self, data: pd.Series, mask: pd.Series, max_valid_time: float):
        # Initialize vectors and indices for previous and next value.
        prev_value = np.zeros(len(data))
        next_value = np.zeros(len(data))
        prev_index = -max_valid_time*self.frequency - 1  # Such that it is "too long ago".
        next_index = np.nan
        last_valid_value = np.nan

        # Loop through the vector
        for i, (mask_now, mask_succ, now, succ) in enumerate(zip(mask.iloc[:-1], mask.iloc[1:],
                                                                 data.iloc[:-1], data.iloc[1:])):
            if mask_now:
                prev_index = i
                prev_value[i] = now
                last_valid_value = now
            else:
                if i - prev_index <= max_valid_time*self.frequency:
                    prev_value[i] = last_valid_value
                else:
                    prev_value[i] = np.nan
            if not next_index > i:
                # The following if is not really needed, because this can be done with the `next`
                # statement in the `else` code. However, because it happens so often that the next
                # sample is valid, is saves us time if we do not need to perform the full .iloc
                # method. Hence, it is worth checking if the next measurement is valid.
                if mask_succ:
                    next_index = i + 1
                    next_value[i] = succ
                else:
                    next_index = next((j+i+2 for j, value in enumerate(mask.iloc[i+2:]) if value),
                                      np.nan)
                    if np.isnan(next_index) or next_index - i > max_valid_time*self.frequency:
                        next_value[i] = np.nan
                    else:
                        next_value[i] = data.iat[next_index]
            else:
                next_value[i] = next_value[i-1]

        # Set the last row
        if mask.iat[-1]:
            prev_value[-1] = data.iat[-1]
        else:
            prev_value[-1] = prev_value[-2] if len(data) > 1 else np.nan
        next_value[-1] = np.nan

        return prev_value, next_value

    def to_hdf(self, path: str, complevel: int = 4) -> None:
        """ Save the data to an HDF5 file.

        :param path: The path to the HDF5 file.
        :param complevel: Compression level, default=4.
        """
        self.data.to_hdf(path, "Data", mode="w", complevel=complevel)
        if self.targets:
            targets = pd.concat(self.targets, sort=False)
            targets.to_hdf(path, "Targets", mode="a", complevel=complevel)

    def read_hdf(self, path: str) -> None:
        """ Load the data (and targets) from a HDF5 file.

        :param path: The path to the HDF5 file.
        """
        with pd.HDFStore(path, 'r') as hdf:
            self.data = hdf["Data"]
            if "Targets" in hdf:
                self.targets = self._big_target_df_to_list(hdf["Targets"])
