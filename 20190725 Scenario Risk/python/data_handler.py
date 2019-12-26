""" Class for general functions for handling the data

Creation date: 2019 12 09
Author(s): Erwin de Gelder

Modifications:
2019 12 15 Group data by target id to create separate dataframe for each target.
2019 12 19 Provide the option to give the frequency.
2019 12 26 Save targets too.
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
        """ Create separate dataframes for all targets. """
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
        targets = pd.concat(dfsubs)
        targets = list(targets.groupby("id"))[1:]  # Skip the first as it refers to id=0 (no target)
        targets = [target[1] for target in targets]
        targets = [target.sort_index() for target in targets]
        return targets

    def to_hdf(self, path: str, complevel: int = 4) -> None:
        """ Save the data to an HDF5 file.

        :param path: The path to the HDF5 file.
        :param complevel: Compression level, default=4.
        """
        self.data.to_hdf(path, "Data", mode="w", complevel=complevel)
        if self.targets:
            targets = pd.concat(self.targets)
            targets.to_hdf(path, "Targets", mode="a", complevel=complevel)

    def read_hdf(self, path: str) -> None:
        """ Load the data (and targets) from a HDF5 file.

        :param path: The path to the HDF5 file.
        """
        hdf = pd.HDFStore(path, 'r')
        self.data = hdf["Data"]
        if "Targets" in hdf:
            targets = hdf["Targets"]
            targets = list(targets.groupby("id"))
            self.targets = [target[1] for target in targets]
