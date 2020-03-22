""" Class for constructing multiple n-grams that have a similar structure

Creation date: 2019 11 27
Author(s): Erwin de Gelder

Modifications:
2019 12 31 Change the way the n-grams are stored.
2019 01 23 Add last row to n-gram. Otherwise, last item will be missed!
2020 03 22 Add functionality to find start and end indices during a time frame.
"""

import os
from typing import Any, Iterable, Tuple
import numpy as np
import pandas as pd


class NGram:
    """ Class for constructing multiple n-grams.

    If the fieldnames contain the fieldname "id", it is assumed that there are
    multiple n-grams provided. Hence, the attribute `ngrams` will be used and
    the attribute `ngram` is not used. If the fieldname "id" is not provided,
    the attributebute `ngrams` is not used and the attribute `ngram` contains
    one dataframe.

    Attributes:
        fieldnames (Iterable[str]): Names of the fields of the n-gram.
        metafields (Iterable[Tuple[str, Any]]): Names and types of the metadata.
        ngrams (List[pd.DataFrame]): List of n-grams (if there are multiple).
        ngram (pd.DataFrame): The n-gram (if there is only one).
        metadata (pd.DataFrame): Metadata of the ngrams.
    """
    def __init__(self, fieldnames: Iterable[str], metafields: Iterable[Tuple[str, Any]]):
        self.fieldnames = fieldnames
        self.metafields = metafields
        self.ngrams = []
        self.ngram = pd.DataFrame()
        self.metadata = pd.DataFrame(columns=[name for name, _ in metafields])

        # For the fieldnames, the field "index" is preserved, so that one cannot
        # be used. If it is used, an error will be raised as to avoid weird
        # behavior later on.
        if "index" in self.fieldnames:
            raise ValueError("The fieldnames cannot contain a name 'index', because that name" +
                             " is preserved")

        # For the metadata, the field "data" is preserved, so that one cannot be
        # used. If it is used, an error will be raised as to avoid weird
        # behavior later on.
        for metafield in self.metafields:
            if metafield[0] == "data":
                raise ValueError("The metafields cannot contain a name 'data', because that name" +
                                 " is preserved.")

    def add_ngram(self, data: pd.DataFrame, **kwargs):
        """ Create an n-gram and add it to the list of n-grams.

        :param data: The dataframe that contains the events.
        :param kwargs: The metadata that has to be provided.
        """
        # Create the metadata dictionary.
        metadata = dict()
        for metafield in self.metafields:
            if metafield[0] not in kwargs:
                raise KeyError("Metadata field '{}' is not provided.".format(metafield[0]))
            if not isinstance(kwargs[metafield[0]], metafield[1]):
                raise TypeError("Metadata field '{}' is op type '{}' but should be of type '{}'".
                                format(metafield[0], type(kwargs[metafield[0]]), metafield[1]))
            metadata[metafield[0]] = kwargs[metafield[0]]
        self.metadata = self.metadata.append(metadata, ignore_index=True)

        # Add the n-gram to the list of n-grams.
        if "id" in self.fieldnames:
            self.ngrams.append(data)
        else:
            self.ngram = data

    def ngram_from_data(self, data: pd.DataFrame, **kwargs):
        """ Convert the data such that it can be used to create an n-gram.

        Two things are done:
        1. All fields that are not in self.fieldnames are removed (except for
           the index).
        2. If row i+1 equals row i (except for the index), the row is removed.

        :param data: The dataframe that is used to get the events from.
        :param kwargs: The metadata that has to be provided.
        """
        # Only use the relevant data.
        my_df = data[self.fieldnames]

        # Get the indices of the rows that are different from its previous rows.
        indices = [my_df.index[0]]
        for index, row_update, row in zip(my_df.index[1:], my_df.iloc[1:].itertuples(index=False),
                                          my_df.iloc[:-1].itertuples(index=False)):
            if row_update != row:
                indices.append(index)

        # Add the last row if it is not in yet.
        if indices[-1] != my_df.index[-1]:
            indices.append(my_df.index[-1])

        # Add the n-gram.
        self.add_ngram(my_df.loc[indices], **kwargs)

    def sort_ngrams(self, fieldname: str, ascending: bool = True) -> None:
        """ Sort the n-grams based on the value of the given fieldname.

        :param fieldname: The name of the metafield that is used for sorting.
        :param ascending: Whether to sort ascedingly (default) or not.
        """
        # This only makes sense if multiple n-grams are used.
        if "id" in self.fieldnames:
            self.metadata.sort_values(fieldname, inplace=True, ascending=ascending)
            self.ngrams = [self.ngrams[i] for i in self.metadata.index]

    def start_and_end_indices(self, fieldname: str, i_ngram: int = None, i_start: float = None,
                              i_end: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Find the start and end indices of parts that do not change.

        Suppose, the Series of <fieldname> look like (a, a, b, b, a) for indices
        (0, 1, 2, 3, 4). Then we have the parts (a, a), (b, b), and (a) that do
        not change. Hence, the starting indices are [0, 2, 4], and the end
        indices are [1, 3, 4].

        :param fieldname: The name of the series that will be examined.
        :param i_ngram: The i-th n-gram if this object contains more n-grams.
        :param i_start: Optional, the starting index to start looking.
        :param i_end: Optional, the end index to stop looking.
        :return: The starting and ending indices, e.g., ([0, 2, 4], [1, 3, 4]).
        """
        # Select the signal that we are examining.
        if i_ngram is None:
            signal = self.ngram[fieldname]
        else:
            signal = self.ngrams[i_ngram][fieldname]

        # Get starting and end index.
        if i_start is None:
            i_start = signal.index[0]
        else:
            i_start = signal.index[signal.index.get_loc(i_start, method="pad")]
            current_value = signal.loc[i_start]
            for i, value in signal.loc[:i_start][::-1].iteritems():
                if not value == current_value:
                    break
                i_start = i
        if i_end is None:
            i_end = signal.index[-1]
        else:
            i_end = signal.index[signal.index.get_loc(i_end, method="backfill") - 1]

        # Search for changes.
        start = [i_start]
        end = []
        current_value = signal.loc[i_start]
        for i, value in signal.loc[i_start:].iteritems():
            if not value == current_value:
                end.append(i)
                if i >= i_end:
                    break
                current_value = value
                start.append(i)
        return np.array(start), np.array(end)

    def to_hdf(self, path: str, name: str, mode="a", complevel: int = 4) -> None:
        """ Save the n-grams to an HDF5 file.

        :param path: The path to the HDF5 file.
        :param name: Name of the data.
        :param mode: {'a', 'w'}, whether to write a new file or append file.
        :param complevel: Compression level, default=4.
        """
        self.metadata.to_hdf(path, "{:s}/Metadata".format(name), mode=mode, complevel=complevel)
        if "id" in self.fieldnames:
            ngrams = pd.concat(self.ngrams, sort=False)
            ngrams.to_hdf(path, "{:s}/nGrams".format(name), mode="a", complevel=complevel)
        else:
            self.ngram.to_hdf(path, "{:s}/nGram".format(name), mode="a", complevel=complevel)

    def from_hdf(self, path: str, name: str) -> bool:
        """ Read the n-gram data from an HDF5 file.

        :param path: The path to the HDF5 file.
        :param name: Name of the data.
        :return: Whether the data is succesfully loaded or not.
        """
        if not os.path.exists(path):
            return False  # File does not exist.
        with pd.HDFStore(path) as hdf:
            if name not in hdf:
                return False  # Data is not in the HDF file.
            self.metadata = hdf["{:s}/Metadata".format(name)]
            if "id" in self.fieldnames:
                ngrams = hdf["{:s}/nGrams".format(name)]
                ngrams = list(ngrams.groupby("id"))
                self.ngrams = [ngram for _, ngram in ngrams]
            else:
                self.ngram = hdf["{:s}/nGram".format(name)]
            return True
