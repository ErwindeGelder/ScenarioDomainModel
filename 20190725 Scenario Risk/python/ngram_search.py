""" Function for searching for ngrams within n-gram models.

Creation date: 2020 01 12
Author(s): Erwin de Gelder

Modifications:
"""

from typing import NamedTuple, Sequence
import numpy as np
import pandas as pd


class _NGramSearch(NamedTuple):
    """ Result for single n-gram search. """
    is_found: bool
    index: int = 0
    time: float = 0.0


class _StartEnd(NamedTuple):
    """ Result for searching of start of an n-gram. """
    is_found: bool
    t_start: float = None
    t_end: float = None


def check_row(row, dict_tags: dict) -> bool:
    """ Check if a row of a dataframe contains the provided tags.

    Each item of the dictionary needs to contain a list.

    :param row: The row that is obtained through pd.DataFrame.itertuples().
    :param dict_tags: The dictionary of tags.
    :return: Whether the row contains the provided tags.
    """
    for key, tags in dict_tags.items():
        if getattr(row, key) not in tags:
            return False
    return True


def determine_start(ngram: pd.DataFrame, tags: dict, t_start: float = None, t_end: float = None) \
        -> _NGramSearch:
    """ Determine the start for a given item in an n-gram model.

    :param ngram: The n-gram model.
    :param tags: The tags that define the item.
    :param t_start: If given, the start may not be before t_start.
    :param t_end: If given, the start may not be after t_end
    :return: Whether the item is found and, if so, the index and time.
    """
    # Check if the n-gram has data within [tstart, tend].
    if t_start is not None and t_end is not None:
        if t_start > ngram.index[-1] or t_end < ngram.index[0]:
            return _NGramSearch(False)

    # Determine the index at which we can start searching.
    i_start = 0
    if t_start is not None and t_start > ngram.index[0]:
        i_start = ngram.index.get_loc(t_start, method='pad')
        ngram = ngram.iloc[i_start:]

    # Search for a match.
    for i, row in enumerate(ngram.itertuples(), start=i_start):
        if t_end is not None and row.Index > t_end:
            break
        if check_row(row, tags):
            if t_start is not None and t_start > row.Index:
                return _NGramSearch(True, i, t_start)
            return _NGramSearch(True, i, row.Index)

    # No match is found.
    return _NGramSearch(False)


def determine_end(ngram: pd.DataFrame, tags: dict, istart: int, t_end: float = None) \
        -> _NGramSearch:
    """ Determine the end of an item in an n-gram model.

    It might be that the end is not found, but this is only in the hypothetical
    case that the index refers to the last item of the n-gram model.

    :param ngram: The n-gram model.
    :param tags: The tags that define the item.
    :param istart: The starting index of the item.
    :param t_end: If provided, the end time may not exceed t_end.
    :return: Whether an end time is found and, if so, its index and time.
    """
    row, i = None, 0
    for i, row in enumerate(ngram.iloc[istart + 1:].itertuples(), start=istart + 1):
        if not check_row(row, tags):
            break
        if t_end is not None and row.Index > t_end:
            break
    if row is None:  # This might happen if istart+1 == len(ngram).
        return _NGramSearch(False)
    if t_end is not None and t_end < row.Index:
        return _NGramSearch(True, i, t_end)
    return _NGramSearch(True, i, row.Index)


def determine_start_end(ngram: pd.DataFrame, tags: dict, previous_search: _StartEnd = None) \
        -> _StartEnd:
    """ Determine the start and end of an item in an n-gram model.

    :param ngram: The n-gram model.
    :param tags: The tags that define the item.
    :param previous_search: If provided, the search will be in between the
        provided start and end time.
    :return: Whether a match is found and, if so, its start and end time.bn
    """
    if previous_search is None:
        previous_search = _StartEnd(False)
    start = determine_start(ngram, tags, previous_search.t_start, previous_search.t_end)
    if not start.is_found:
        return _StartEnd(False)
    end = determine_end(ngram, tags, start.index, previous_search.t_end)
    if not end.is_found:
        return _StartEnd(False)
    return _StartEnd(True, start.time, end.time)


def find_part_of_sequence(ngrams: Sequence[pd.DataFrame], tags: Sequence[dict],
                          t_start: float = None, force_start: bool = False) -> _StartEnd:
    """ Find match at which each item is found in its corresponding n-gram model.

    :param ngrams: The n-gram models.
    :param tags: The items, one item for each n-gram model.
    :param t_start: If provided, the starting time should be after t_start.
    :param force_start: If True, the start should be at t_start.
    :return: Whether a match is found and, if so, the start and end time.
    """
    # Check for the first item of first n-gram model.
    searches = np.zeros(len(ngrams), dtype=_StartEnd)
    searches[0] = _StartEnd(True, t_start, ngrams[0].index[-1])
    level = 0
    while True:
        # Four possible results:
        # 1. Tag found and not at highest level.
        # 2. Tag found and at highest level, so return True.
        # 3. Tag not found and not at lowest level, so go one level up and start search from the
        #    previous end.
        # 4. Tag not found and at lowest level, so return False.
        search = determine_start_end(ngrams[level], tags[level], previous_search=searches[level])
        if search.is_found:  # Possibility 1 or 2.
            if force_start and search.t_start > t_start:
                return _StartEnd(False)
            level += 1
            if level < len(ngrams):  # Possibility 1.
                searches[level] = search
            else:  # Possibility 2.
                return _StartEnd(True, search.t_start, search.t_end)
        else:  # Possibility 3 or 4.
            level -= 1
            if level >= 0:  # Possibility 3.
                # We need to go one level up and start searching from the previous end
                # to see if we can find a new match. However, it might be possible that
                # the new window has length 0. In that case, we need to go one level up.
                # This might continue until we reach the lowest level. In that case, we
                # will not find a match, so we can return a False
                while searches[level + 1].t_end >= searches[level].t_end:
                    level -= 1
                    if level < 0:
                        return _StartEnd(False)
                searches[level] = _StartEnd(False, searches[level + 1].t_end, searches[level].t_end)
            else:  # Possibility 4.
                return _StartEnd(False)


def find_sequence(ngrams: Sequence[pd.DataFrame], tags: Sequence[Sequence[dict]],
                  t_start: float = None) -> _StartEnd:
    """ Find the n-grams in the n-gram models.

    :param ngrams: The n-gram models.
    :param tags: The tags that define the items. List of n-grams.
    :param t_start: If provided, the result should start after t_start.
    :return: Whether a match is found and, if so, its start and end time.
    """
    # Check for the first item for each n-gram model.
    search = find_part_of_sequence(ngrams, [tag[0] for tag in tags], t_start)
    if not search.is_found:
        return _StartEnd(False)
    t_start = search.t_start

    # Go through remaining steps.
    for j in range(1, len(tags[0])):
        search = find_part_of_sequence(ngrams, [tag[j] for tag in tags], search.t_end,
                                       force_start=True)
        if not search.is_found:
            return _StartEnd(False)
    return _StartEnd(True, t_start, search.t_end)
