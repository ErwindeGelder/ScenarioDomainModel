""" Convert mat files to HDF5 files with pandas dataframes

Creation date: 2019 10 22
Author(s): Erwin de Gelder

Modifications:
2019 11 01 Add arguments and the possibility to convert one single file.
2019 12 26 Use the data handler and fix targets that get lost in blind spot.
2019 12 30 Remove information in trackers as it will be outdated and should not be used.
2020 02 24 Print the total time duration.
"""

import argparse
import glob
import multiprocessing as mp
import os
import time
from tqdm import tqdm
from mat2df import Mat2DF
from data_handler import DataHandler
from target_gluer import merge_targets


PARSER = argparse.ArgumentParser(description="Convert .mat data to .hdf5")
PARSER.add_argument('-complevel', default=4, type=int, choices=range(10),
                    help="Compression level, default=4")
PARSER.add_argument('-matfolder', default=os.path.join("data", "0_mat_files"), type=str,
                    help="Folder to get .mat data from")
PARSER.add_argument('-outfolder', default=os.path.join("data", "1_hdf5"), type=str,
                    help="Folder to write the data to")
PARSER.add_argument('-file', default=None, type=str, help="If not all files, select single file")
ARGS = PARSER.parse_args()


# Define the columns that are to be removed.
REMOVE = ["navposllh_iTOW", "navposllh_hMSL", "navposllh_hAcc",
          "navposllh_vAcc", "navposllh_height"]
for i in range(2):
    for key in ["length", "type", "width"]:
        REMOVE.append("lines_{:d}_{:s}".format(i, key))
for i in range(8):
    for key in ["vel_theta", "accel_theta", "vel_y", "accel_y"]:
        REMOVE.append("wm_targets_{:d}_{:s}".format(i, key))

# Define the columns that are to be renamed.
RENAME = {"ego_lonVel": "Host_vx", "ego_yawrate": "Host_yawrate", "ego_lonAcc": "Host_ax",
          "ego_delta": "Host_theta_steeringwheel", "navposllh_lon": "gps_lon",
          "navposllh_lat": "gps_lat"}
for i in range(2):
    for key, value in dict(y="c0", confidence="quality").items():
        RENAME["lines_{:d}_{:s}".format(i, key)] = "lines_{:d}_{:s}".format(i, value)
for i in range(8):
    for key, value in dict(id="id", age="age", pose_x="dx", pose_y="dy", pose_theta="theta",
                           vel_x="vx", accel_x="ax", probability_of_existence="prob").items():
        RENAME["wm_targets_{:d}_{:s}".format(i, key)] = "Target_{:d}_{:s}".format(i, value)


def conv_mat_file(matfile: str) -> None:
    """ Convert a single matfile.

    :param matfile: Name of the file that has to be converted.
    """
    conv = Mat2DF(matfile)
    try:
        conv.convert()
    except Exception as exception:
        print("Error at matfile: {:s}".format(matfile))
        raise exception
    conv.data = conv.data.drop(columns=REMOVE)
    conv.data = conv.data.rename(columns=RENAME)
    fix_gps(conv)
    conv.data["Time"] = conv.data.index.values
    conv.data.index = (conv.data.index - conv.data.index[0]) / 1e9

    # Merge targets that get lost in the blind spot.
    data_handler = DataHandler(conv.data)
    merge_targets(data_handler)

    # Remove the data in the trackers, since it is outdated anyway.
    signals = data_handler.get_target_signals()
    remove = []
    for j in range(data_handler.n_trackers):
        remove += [data_handler.target_signal(j, signal) for signal in signals]
    data_handler.data = data_handler.data.drop(columns=remove)

    # Safe to HDF file.
    filename = os.path.join("data", "1_hdf5",
                            "{:s}.hdf5".format(os.path.splitext(os.path.basename(matfile))[0]))
    data_handler.to_hdf(filename, complevel=ARGS.complevel)


def fix_gps(conv: Mat2DF) -> None:
    """ Fix the GPS data.

    The fix works very simple: If the dataframe has a column "position_x", then
    the data in "position_x" and "position_y" is written to the columns
    "Host_gps_lon" and "Host_gps_lat", respectively. Next, the columns
    "position_x" and "position_y" are removed.

    :param conv: The converter object that has the field 'data'.
    """
    if "position_x" in conv.data.keys():
        conv.data["Host_gps_lon"] = conv.data["position_x"]
        conv.data["Host_gps_lat"] = conv.data["position_y"]
        conv.data = conv.data.drop(columns=["position_x", "position_y"])


if __name__ == "__main__":
    TSTART = time.time()
    if ARGS.file is None:
        MATFILES = glob.glob(os.path.join(ARGS.matfolder, "*.mat"))
        with mp.Pool(processes=4) as POOL:
            for _ in tqdm(POOL.imap_unordered(conv_mat_file, MATFILES), total=len(MATFILES)):
                pass
    else:
        conv_mat_file(os.path.join(ARGS.matfolder, ARGS.file))
    print("Total elapsed time: {:.2f} s".format(time.time() - TSTART))
