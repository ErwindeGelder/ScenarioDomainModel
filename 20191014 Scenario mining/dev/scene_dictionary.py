"""
Creating and maintaining a dictionary of observed scenes

Author
------
Erwin de Gelder

Creation
--------
14 Oct 2019

To do
-----

Modifications
-------------
"""


import os
import numpy as np
import pandas as pd
from typing import List
from options import Options


class SceneDictionaryOptions(Options):
    """ Default options for the SceneDictionary """
    data_folder: str = os.path.join('..', 'data')
    raw_data_folder: str = '00_raw_data'
    dictionary_folder: str = '01_activity_dictionary'
    dictionary_filename_prefix: str = 'scene_dictionary'
    relative_positions: dict = {'longitudinal': ['x<-50',
                                                 '-50<=x<-30',
                                                 '-30<=x<-10',
                                                 '-10<=x<0',
                                                 '0<=x<=10',
                                                 '10<x<=30',
                                                 '30<x<=50',
                                                 'x>50'],
                                'lateral': ['left_lane', 'same_lane', 'right_lane']}
    host_columns: List[str] = ['host_lateral', 'host_longitudinal']
    target_columns: List[str] = ['lateral', 'longitudinal', 'relative_position_longitudinal',
                                 'relative_position_lateral', 'velocity']
    n_targets = 8
    debug: bool = True


class SceneDictionaryParameters(Options):
    """ Parameters for the SceneDictionary """
    dictionary_filename: str = None
    dictionary_columns: List = None


class SceneDictionary:
    """ Creating and maintaining a dictionary of observed scenes

    Attributes:
        options(SceneDictionaryOptions): Configurations parameters.
        parms(SceneDictionaryParameters): All kinds of parameters.
        dictionary(pd.DataFrame): Dataframe that contains the dictionary.
    """
    def __init__(self, options: SceneDictionaryOptions = None):
        self.options = SceneDictionaryOptions() if options is None else options
        self.parms = SceneDictionaryParameters()

        # Get names right.
        self.parms.dictionary_filename = os.path.join(
            self.options.data_folder, self.options.dictionary_folder,
            "{:s}_{:d}x{:d}.csv".format(self.options.dictionary_filename_prefix,
                                        len(self.options.relative_positions['longitudinal']),
                                        len(self.options.relative_positions['lateral'])))

        # Define all columns of the dictionary.
        self.parms.dictionary_columns = self.options.host_columns + \
            ["target_{:d}_{:s}".format(i, name) for i in range(self.options.n_targets)
             for name in self.options.target_columns]

        # Initialize the dictionary.
        if os.path.exists(self.parms.dictionary_filename):
            self.dictionary = pd.read_pickle(self.parms.dictionary_filename)
        else:
            self.dictionary = pd.DataFrame(columns=self.parms.dictionary_columns)

    def run(self, filename: str) -> None:
        """ Update the dictionary.

        Update the dictionary based on the given filename. If filename='all',
        then all files in the input folder are processed.

        :param filename: The name of the file.
        """
        self.dictionary_update(filename)

    def dictionary_update(self, filename: str) -> None:
        """ Update the dictionary based on one file.

        :param filename:
        """
        if self.options.debug:
            print("\n** DICTIONARY UPDATE **")
        reduced_dataset = self.load_dataset(filename)
        if self.options.debug:
            print("\t03. Dictionary construction of the corrected dataframe ", end="")
        unique_entries = reduced_dataset[self.parms.dictionary_columns].drop_duplicates()
        unique_entries.reset_index(inplace=True, drop=True)
        if self.options.debug:
            print("(shape={})..".format(unique_entries.shape), end="")

        self.dictionary = unique_entries.copy()
        return

        n_add = 0
        n_skip = 0
        if self.options.debug:
            print("    ", end="")

        for i, entry in enumerate(unique_entries.itertuples()):
            if self.options.debug:
                print("*" if (i+1) % 10 == 0 else ".", end="")

            # Is the entry already present in the dictionary?
            add_entry = False
            if len(self.dictionary) == 0:
                add_entry = True
            elif np.max(np.sum(self.dictionary[entry.Index] == entry, axis=1)) < len(entry):
                add_entry = True

            if add_entry:
                entry['index'] = len(self.dictionary)
                entry['n_objects]'] = sum([entry['target_{:d}_lateral'.format(j)] != ''
                                           for j in range(self.options.n_targets)])
                n_add += 1
                entry_dict = {k: entry[k] for k in self.parms.dictionary_columns}
                self.dictionary = self.dictionary.append(entry_dict, ignore_index=True)
            else:
                n_skip += 1

        if self.options.debug:
            print("\n\t\tDataset processed, new entries={:d}, existing entries={:d}"
                  .format(n_add, n_skip))
            print("\t\tupdated dictionary shape={}".format(self.dictionary.shape))

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """ Load a specified dataframe.

        :param filename: The name of the file.
        :return: The dataframe with the data.
        """
        # Load the dataset.
        if self.options.debug:
            print("\t01. Loading {:s}..".format(filename), end="")
        with pd.HDFStore(os.path.join(self.options.data_folder, self.options.raw_data_folder,
                                      filename)) as hdf_store:
            tagged_dataset = hdf_store.get('df')
        tagged_dataset.reset_index(inplace=True, drop=True)
        if self.options.debug:
            print("OK, shape={}".format(tagged_dataset.shape))

        # Correction of dataset based on the grid-size.
        if self.options.debug:
            print("\t02. Dataset correction based on dictionary grid-size..", end="")
        for i in range(self.options.n_targets):
            tagged_dataset.loc[
                ~tagged_dataset["target_{:d}_relative_position_longitudinal".format(i)].isin(
                    self.options.relative_positions["longitudinal"]) |
                ~tagged_dataset["target_{:d}_relative_position_lateral".format(i)].isin(
                    self.options.relative_positions["lateral"]),
                ["target_{:d}_{:s}".format(i, name) for name in self.options.target_columns]
            ] = ""

        if self.options.debug:
            print("OK")

        return tagged_dataset
