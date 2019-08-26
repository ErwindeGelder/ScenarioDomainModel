""" Load the ERP2017 data and add extra information

Creation data: 2019 08 06
Author(s): Erwin de Gelder

Modifications:
"""


import os
from glob import glob
import time
import pandas as pd
import matplotlib.pyplot as plt
from activity_detector import ActivityDetector


DATA_FOLDER = os.path.join('..', '..', '..', '..', '..', 'TNO',
                           'Manders, T.M.C. (Jeroen) - ERP2017Release')
DATAFILES = glob(os.path.join(DATA_FOLDER, 'Data', '*.hdf5'))
SIGNAL_DESCRIPTIONS = pd.read_csv(os.path.join(DATA_FOLDER, 'signal_descriptions.csv'),
                                  index_col=0).squeeze().to_dict()
DATAFRAME = pd.read_hdf(DATAFILES[0])  # type: pd.DataFrame
ACTIVITY_DETECTOR = ActivityDetector(DATAFRAME)
DATAFRAME = ACTIVITY_DETECTOR.get_all_data()

tstart = time.time()
EVENTS = ACTIVITY_DETECTOR.long_activities_host()
print("Time: {}".format(time.time() - tstart))

print(EVENTS)
plt.plot(DATAFRAME.index, DATAFRAME["Host_vx"])
COLORS = [("r" if event == "a" else ("b" if event == "d" else "g")) for _, event in EVENTS]
plt.vlines([time for time, _ in EVENTS], plt.ylim()[0], plt.ylim()[1], colors=COLORS)
plt.show()
