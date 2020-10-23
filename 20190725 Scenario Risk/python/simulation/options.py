""" Class that does not allow for creating new attributes.

Creation date: 2019 06 26
Author(s): Erwin de Gelder

Modifications:
"""


class Options:
    """ Contain options of objects.

    The goal of this class is to create objects that does not allow for creating
    new attributes. In this way, this cannot be done accidentily, as this can
    generally cause very weird errors.
    """
    __isfrozen = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                self.__getattribute__(key)
            except AttributeError:
                raise KeyError("Option '{:s}' is not a valid option.".format(key))
            self.__setattr__(key, value)

        # Make sure that no new attributes are created outside the __init__ function.
        self.freeze()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("{} is a frozen class".format(self))
        object.__setattr__(self, key, value)

    def freeze(self) -> None:
        """ Freeze the attributes, meaning that it is not possible to add attributes. """
        self.__isfrozen = True

    def unfreeze(self) -> None:
        """ Unfreeze the attributes, meaning that it is possible to add attributes. """
        self.__isfrozen = False
