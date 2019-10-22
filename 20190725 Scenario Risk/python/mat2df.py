""" Convert mat files to HDF5 files with pandas dataframes

Creation data: 2019 10 22
Author(s): Erwin de Gelder

Modifications:
"""

import os
from typing import List
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.interpolate as sinterp
from options import Options


class Mat2DFOptions(Options):
    """ Options for converting a mat file to a pandas dataframe. """
    topics: List[str] = ["ego", "lines", "navposllh", "wm_targets"]
    topic_start_end_time: str = "ego"
    fps: int = 100


class Mat2DF:
    """ Class for converting a mat file to a pandas dataframe.

    Attributes:
        data: A pandas dataframe with the data.
        options: Configuration options for converting the matfile.
        mat_path: Path to the .mat file.
    """
    def __init__(self, mat_path: str, options: Mat2DFOptions = None):
        self.options = Mat2DFOptions() if options is None else options
        self.mat_path = mat_path
        self.data = pd.DataFrame()

    def convert(self) -> None:
        """ Convert the matfile to a pandas dataframe. """
        # Load the mat file.
        matfile = sio.loadmat(self.mat_path, struct_as_record=False, squeeze_me=True)

        # Initialize the dataframe.
        time_vec = getattr(matfile["data"], self.options.topic_start_end_time).time
        self.data = pd.DataFrame(index=range(time_vec[0], time_vec[-1],
                                             1000000000//self.options.fps))

        # Loop through the topics and add them to the dataframe.
        for topic in self.options.topics:
            # Obtain the data
            mat_data = getattr(matfile["data"], topic).data
            nsubtopics = 1
            if len(mat_data.shape) == 3:
                nsubtopics = mat_data.shape[2]
                mat_data = np.reshape(mat_data, [mat_data.shape[0], mat_data.shape[1]*nsubtopics])

            # Obtain the fieldnames.
            fieldnames = getattr(getattr(matfile["idx"], topic), "_fieldnames")
            index = [getattr(getattr(matfile["idx"], topic), attr) for attr in fieldnames]
            if nsubtopics == 1:
                sorted_fieldnames = ["{:s}_{:s}".format(topic, fieldnames[i])
                                     for i in np.argsort(index)]
            else:
                sorted_fieldnames = ["{:s}_{:d}_{:s}".format(topic, j, fieldnames[i])
                                     for i in np.argsort(index) for j in range(nsubtopics)]

            # Create a dataframe with the data from the topic and add it to the data.
            interpolator = sinterp.interp1d(getattr(matfile["data"], topic).time, mat_data, axis=0,
                                            kind="nearest", fill_value=0, bounds_error=False)
            df_topic = pd.DataFrame(index=self.data.index, columns=sorted_fieldnames,
                                    data=interpolator(self.data.index))
            self.data = pd.concat([self.data, df_topic], axis=1)

    def save2hdf5(self, filename: str, complevel: int = 4) -> None:
        """ Save the dataframe to an HDF5 file.

        :param filename: The filename of the to-be-saved HDF5 file.
        :param complevel: Compression level, default=4.
        """
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        self.data.to_hdf(filename, 'Data', mode='w', complevel=complevel)
