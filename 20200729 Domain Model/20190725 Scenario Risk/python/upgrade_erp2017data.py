""" Load the ERP2017 data and add extra information

Creation date: 2019 08 06
Author(s): Erwin de Gelder

Modifications:
2019 08 28 Automatically save updated dataframe. Loop through all dataframes.
2019 10 22 Use multiprocessing to speed up the analysis.
2019 11 01 Add possibility to mark highway data.
2019 12 30 Use updated version of the activity detector.
2020 02 24 Print the total time duration.
2020 04 26 Add tag for vehicles that are somewhat near.
"""


import argparse
from glob import glob
import os
import multiprocessing as mp
import time
from tqdm import tqdm
from activity_detector import ActivityDetector
from mark_highway import HighwayMarker


PARSER = argparse.ArgumentParser(description="Read ERP data and add information.")
PARSER.add_argument('-complevel', default=4, type=int, choices=range(10),
                    help="Compression level, default=4")
PARSER.add_argument('-folder', default=os.path.join("data", "1_hdf5"), type=str,
                    help="Folder to write new data to")
PARSER.add_argument('--hostactivities', help="Detect host activities", action="store_true")
PARSER.add_argument('--targetactivities', help="Detect host activities", action="store_true")
PARSER.add_argument('--markhighway', help="Mark highway data", action="store_true")
PARSER.add_argument('--targetstates', help="Create tags describing the target states",
                    action="store_true")
PARSER.add_argument('--targetnear', help="Mark near targets", action="store_true")
PARSER.add_argument('--everything', help="Do all possible processing", action="store_true")
PARSER.add_argument('-file', default=None, type=str, help="If not all files, select single file")
ARGS = PARSER.parse_args()


def process_file(datafile: str) -> None:
    """ Process an HDF5 file.

    :param datafile: Path of the to-be-processed file.
    """
    try:
        activity_detector = ActivityDetector(datafile)
        if ARGS.hostactivities or ARGS.everything:
            activity_detector.set_lon_activities_host()
            activity_detector.set_lat_activities_host()
        if ARGS.targetactivities or ARGS.everything:
            activity_detector.set_target_activities()
        if ARGS.targetstates or ARGS.everything:
            activity_detector.set_states_targets()
            activity_detector.set_lead_vehicle()
        if ARGS.targetnear or ARGS.everything:
            activity_detector.set_near_targets()
        if ARGS.markhighway or ARGS.everything:
            highway_marker = HighwayMarker(activity_detector.data)
            highway_marker.mark_highway()

        activity_detector.to_hdf(os.path.join(ARGS.folder, os.path.basename(datafile)),
                                 complevel=ARGS.complevel)
    except Exception as exception:
        print("Error at file: {:s}".format(datafile))
        raise exception


if __name__ == "__main__":
    TSTART = time.time()
    if not os.path.exists(ARGS.folder):
        print("Provided folder '{:s}' does not exist.".format(ARGS.folder))
        exit()
    if ARGS.file is None:
        DATAFILES = glob(os.path.join(ARGS.folder, '*.hdf5'))
        with mp.Pool(processes=4) as POOL:
            for _ in tqdm(POOL.imap_unordered(process_file, DATAFILES), total=len(DATAFILES)):
                pass
    else:
        if not os.path.exists(os.path.join(ARGS.folder, ARGS.file)):
            print("Provided file '{:s}' does not exist.".format(os.path.join(ARGS.folder,
                                                                             ARGS.file)))
            exit()
        process_file(os.path.join(ARGS.folder, ARGS.file))
