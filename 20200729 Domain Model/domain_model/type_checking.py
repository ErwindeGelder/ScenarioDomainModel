""" Functions for type checking

Creation date: 2019 05 22
Author(s): Erwin de Gelder

Modifications:
"""

from typing import List


def check_for_type(input_name, var_to_check, required_type):
    """ Check if variable is of a certain type

    :param input_name: The name of the variable that is checked as a string.
    :param var_to_check: The actual variable that will be checked.
    :param required_type: The required type of the variable that will be checked.
    """
    if not isinstance(var_to_check, required_type):
        raise TypeError("Input '{0}' should be of type {1} but is of type {2}.".
                        format(input_name, required_type, type(var_to_check)))


def check_for_list(input_name, var_to_check, required_type, can_be_none=True, at_least_one=False):
    """ Check if variable is a list containing elements of a certain type

    :param input_name: The name of the variable that is checked as a string.
    :param var_to_check: The actual variable that will be checked.
    :param required_type: The required type of each of the elements of the list.
    :param can_be_none: By default, if the variable to check is None, no error is raised.
    :param at_least_one: By default, the list can be empty. Set to True if at least one element.
    """

    if var_to_check is not None:
        check_for_type(input_name, var_to_check, List)
        if at_least_one and not var_to_check:
            raise ValueError("Input '{0}' should at least contain one value.".format(input_name))
        for element in var_to_check:
            if not isinstance(element, required_type):
                raise TypeError("Items of input '{0}' should be of type ".format(input_name) +
                                "'{0}' but at least one item is of type '{1}'.".
                                format(required_type, type(element)))
    else:
        if not can_be_none:
            raise ValueError("Input '{0}' should be a List with elements of ".format(input_name) +
                             "type {0}, but its value is None.".format(required_type))


def check_for_tuple(input_name, var_to_check, required_types):
    """ Check if variable is a tuple containing the required types

    :param input_name: The name of the variable that is checked as a string.
    :param var_to_check: The actual variable that will be checked.
    :param required_types: The required types of the tuple.
    """

    if not len(var_to_check) == len(required_types):
        raise TypeError("Input '{0}' should be a tuple with length ".format(input_name) +
                        "{:d} but it has length {:d}.".format(len(required_types),
                                                              len(var_to_check)))
    for i, (real, desired) in enumerate(zip(var_to_check, required_types)):
        if not isinstance(real, desired):
            raise TypeError("Item {:d} of '{:s}' should be of type ".format(i, input_name) +
                            "{0} but is of type {1}.".format(desired, type(real)))
