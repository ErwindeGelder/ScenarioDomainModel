""" Using the video time, find the index in the dataframe.

Creation date: 2019 12 08
Author(s): Erwin de Gelder

Modifications:
2020 01 13 Add the option to not print the seconds and the index.
"""

import pandas as pd


def find_neighbours(value: float, data: pd.DataFrame) -> float:
    """ Find index for which the video time is lower than the given value.

    To be more precise, the highest index is found for which the video time is
    lower than the given `value`.

    :param value: The timing of the video.
    :param data: The dataframe that contains the field `video_time`.
    :return: The index in the data.
    """
    exactmatch = data[data.video_time == value]
    if not exactmatch.empty:
        return exactmatch.index[0]
    lowerneighbour_ind = data[data.video_time < value].idxmax()
    return lowerneighbour_ind


def approx_index(minute: int, second: float, data: pd.DataFrame, verbose: bool = True) -> float:
    """ Find the index in the data at the given video time.

    :param minute: The minute in the video.
    :param second: The second in the video.
    :param data: The dataframe that contains the field `video_time`.
    :param verbose: Print the seconds and the index.
    :return: The index in the data.
    """
    iframe = (minute * 60 + second) * 10
    index = find_neighbours(iframe, data)
    if verbose:
        print("Seconds: {:.0f}".format(minute * 60 + second))
        print("Index: {:.2f}".format(index))
    return index
