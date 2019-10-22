"""
Test the dictionary update


Author
------
Erwin de Gelder

Creation
--------
14 Oct 2019

To do
-----

Modifications
-------------
"""

import glob
import os
import time
from dictionary_update_OO import dictionary_update
from scene_dictionary import SceneDictionary

INPUT_FOLDER = os.path.join('..', 'data', '00_raw_data')
OUTPUT_FOLDER = os.path.join('..', 'data', '01_activity_dictionary')
FILELIST = glob.glob(os.path.join(INPUT_FOLDER, '*.hdf5'))
FILENAME = os.path.basename(FILELIST[0])
DICT_UPDATE = dictionary_update(False)
SCENE_DICTIONARY = SceneDictionary()

TSTART = time.time()
DICT_UPDATE.run(FILENAME, False)
print("Time for processing one file: {:.2f} s.".format(time.time() - TSTART))
TSTART = time.time()
SCENE_DICTIONARY.run(FILENAME)
print("Time for processing one file: {:.2f} s.".format(time.time() - TSTART))



# Delete the created dictionaries.
if os.path.exists(os.path.join(OUTPUT_FOLDER, DICT_UPDATE.dictionary_filename)):
    os.remove(os.path.join(OUTPUT_FOLDER, DICT_UPDATE.dictionary_filename))
