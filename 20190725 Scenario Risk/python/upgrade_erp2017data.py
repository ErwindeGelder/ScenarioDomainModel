""" Load the ERP2017 data and add extra information

Creation data: 2019 08 06
Author(s): Erwin de Gelder

Modifications:
2019 08 28 Automatically save updated dataframe. Loop through all dataframes.
"""


import os
from glob import glob
import argparse
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

if not os.path.exists(ARGS.folder):
    os.mkdir(ARGS.folder)
DATAFILES = glob(os.path.join(ARGS.folder, '*.hdf5'))
for datafile in tqdm(DATAFILES):
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
