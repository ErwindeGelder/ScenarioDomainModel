""" Load the ERP2017 data and add extra information

Creation data: 2019 08 06
Author(s): Erwin de Gelder

Modifications:
2019 08 28 Automatically save updated dataframe. Loop through all dataframes.
2019 10 22 Use multiprocessing to speed up the analysis.
"""


import argparse
from glob import glob
import os
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from activity_detector import ActivityDetector


PARSER = argparse.ArgumentParser(description="Read ERP data and add information.")
PARSER.add_argument('-complevel', default=4, type=int, choices=range(10),
                    help="Compression level, default=4")
PARSER.add_argument('-folder', default='data', type=str, help="Folder to write new data to")
PARSER.add_argument('--hostactivities', help="Detect host activities", action="store_true")
PARSER.add_argument('--targetactivities', help="Detect host activities", action="store_true")
ARGS = PARSER.parse_args()


def process_file(datafile: str) -> None:
    """ Process an HDF5 file.

    :param datafile: Path of the to-be-processed file.
    :return:
    """
    dataframe = pd.read_hdf(datafile)  # type: pd.DataFrame
    activity_detector = ActivityDetector(dataframe)
    if ARGS.hostactivities:
        activity_detector.set_lon_activities_host()
        activity_detector.set_lat_activities_host()
    if ARGS.targetactivities:
        for i in range(8):
            activity_detector.set_target_activities(i)
    activity_detector.get_all_data().to_hdf(os.path.join(ARGS.folder, os.path.basename(datafile)),
                                            'Data', mode='w', complevel=ARGS.complevel)


if __name__ == "__main__":
    if not os.path.exists(ARGS.folder):
        os.mkdir(ARGS.folder)
    DATAFILES = glob(os.path.join(ARGS.folder, '*.hdf5'))
    with mp.Pool(processes=4) as POOL:
        for _ in tqdm(POOL.imap_unordered(process_file, DATAFILES), total=len(DATAFILES)):
            pass
