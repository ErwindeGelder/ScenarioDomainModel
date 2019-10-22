""" Convert mat files to HDF5 files with pandas dataframes

Creation data: 2019 10 22
Author(s): Erwin de Gelder

Modifications:
"""


import glob
import multiprocessing as mp
import os
from tqdm import tqdm
from mat2df import Mat2DF


# Define the columns that are to be removed.
REMOVE = ["lines_1_width", "navposllh_iTOW", "navposllh_hMSL", "navposllh_hAcc",
          "navposllh_vAcc"]
for i in range(2):
    for key in ["length", "type"]:
        REMOVE.append("lines_{:d}_{:s}".format(i, key))
for i in range(8):
    for key in ["vel_theta", "accel_theta", "vel_y", "accel_y"]:
        REMOVE.append("wm_targets_{:d}_{:s}".format(i, key))

# Define the columns that are to be renamed.
RENAME = {"ego_lonVel": "Host_vx", "ego_yawrate": "Host_yawrate", "ego_lonAcc": "Host_ax",
          "ego_delta": "Host_theta_steeringwheel", "navposllh_lon": "gps_lon",
          "navposllh_lat": "gps_lat", "navposllh_height": "gps_alt",
          "lines_0_width": "lane_width"}
for i in range(2):
    for key, value in dict(y="c0", confidence="quality").items():
        RENAME["lines_{:d}_{:s}".format(i, key)] = "lines_{:d}_{:s}".format(i, value)
for i in range(8):
    for key, value in dict(id="id", age="age", pose_x="dx", pose_y="dy", pose_theta="theta",
                           vel_x="vx", accel_x="ax", probability_of_existence="prob").items():
        RENAME["wm_targets_{:d}_{:s}".format(i, key)] = "Target_{:d}_{:s}".format(i, value)


def conv_mat_file(matfile: str) -> None:
    """ Convert a single matfile.

    :param matfile:
    """
    conv = Mat2DF(matfile)
    conv.convert()
    conv.data = conv.data.drop(columns=REMOVE)
    conv.data = conv.data.rename(columns=RENAME)
    conv.data.index = (conv.data.index - conv.data.index[0]) / 1e9
    filename = os.path.join("data", "1_hdf5",
                            "{:s}.hdf5".format(os.path.splitext(os.path.basename(matfile))[0]))
    conv.save2hdf5(filename)


if __name__ == "__main__":
    MATFILES = glob.glob(os.path.join("data", "0_mat_files", "*.mat"))
    POOL = mp.Pool(processes=4)
    for _ in tqdm(POOL.imap_unordered(conv_mat_file, MATFILES), total=len(MATFILES)):
        pass
