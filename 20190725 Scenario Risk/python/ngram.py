""" Class for constructing multiple n-grams that have a similar structure

Creation date: 2019 11 27
Author(s): Erwin de Gelder

Modifications:
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Iterable, NamedTuple, Tuple


MyNGram = NamedTuple("ngram", (("meta", dict), ("data", pd.DataFrame)))


class NGram:
    """ Class for constructing multiple n-grams.

    Attributes:

    """
    def __init__(self, fieldnames: Iterable[str], metafields: Iterable[Tuple[str, Any]]):
        self.fieldnames = fieldnames
        self.metafields = metafields
        self.ngrams = []

        # For the fieldnames, the field "index" is preserved, so that one cannot
        # be used. If it is used, an error will be raised as to avoid weird
        # behavior later on.
        if "index" in self.fieldnames:
            raise ValueError("The fieldnames cannot contain a name 'index', because that name"+
                             " is preserved")

        # For the metadata, the field "data" is preserved, so that one cannot be
        # used. If it is used, an error will be raised as to avoid weird
        # behavior later on.
        for metafield in self.metafields:
            if "data" == metafield[0]:
                raise ValueError("The metafields cannot contain a name 'data', because that name" +
                                 " is preserved.")

    def ngram(self, data: pd.DataFrame, **kwargs):
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

        # Add the n-gram to the list of n-grams.
        ngram = MyNGram(meta=metadata, data=data)
        self.ngrams.append(ngram)
        return ngram

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
        for index, row_update, row in zip(my_df.index, my_df.iloc[1:].itertuples(index=False),
                                          my_df.iloc[:-1].itertuples(index=False)):
            if row_update != row:
                indices.append(index)

        # Create and return the n-gram
        return self.ngram(my_df.loc[indices], **kwargs)

    def sort_ngrams(self, fieldname: str, ascending: bool = True) -> None:
        """ Sort the n-grams based on the value of the given fieldname.

        :param fieldname: The name of the metafield that is used for sorting.
        :param ascending: Whether to sort ascedingly (default) or not.
        """
        values = [ngram.meta[fieldname] for ngram in self.ngrams]
        indices = np.argsort(values)
        if not ascending:
            indices = indices[::-1]
        self.ngrams = [self.ngrams[i] for i in indices]

    def to_json(self, filename: str):
        """ Convert the n-grams to a .json file.

        The n-grams are converted to a .json format. Note, however, that the
        .json format is not optimized for speed nor size. The format is meant to
        be readable by experts.

        Later, an optimized version might be added, but this is left as a to do.

        :param filename: The name of the .json file (without ".json").
        """
        ngrams = []
        for ngram in self.ngrams:
            ngram_dict = ngram.meta
            ngram_list = []
            for row in ngram.data.itertuples():
                row_dict = dict(index=row.Index)
                for fieldname in self.fieldnames:
                    row_dict[fieldname] = getattr(row, fieldname)
                ngram_list.append(row_dict)
            ngram_dict["data"] = ngram_list
            ngrams.append(ngram_dict)

        with open(filename, "w") as file:
            json.dump(ngrams, file, indent=4)

    def from_json(self, filename: str):
        """ Reads a .json file and adds it to the list of n-grams.

        :param filename: The name of the .json file (without ".json").
        """
        with open(filename, "r") as file:
            ngrams = json.load(file)
            for ngram in ngrams:
                data = pd.DataFrame(columns=ngram["data"][0].keys())
                for row in ngram["data"]:
                    data = data.append(row, ignore_index=True)
                data.set_index("index")
                ngram.pop("data")
                self.ngram(data, **ngram)

