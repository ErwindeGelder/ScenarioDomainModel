import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import sys

from imports.general_settings import *
from imports.general_functions import *


class dictionary_update(object):

    def __init__(self, print_debug):
        self.file_and_folder_preparation(print_debug)
        self.initialise_dictionary(print_debug)

    # # RENDERING FUNCTIONS

    def make_EGO_IMG_cell(self, i, ego_lateral, ego_longitudinal, debug=False):
        ego_img = np.zeros((EGO_IMG_H, EGO_IMG_W, 3))
        has_issue_longitudinal = False
        has_issue_lateral = False

        if debug:
            print(i, ego_lateral, ego_longitudinal)

        # processing longitutinal information
        if ego_longitudinal == 'a':
            # accelerate
            ego_img += EGO_ACCELERATING_COLOUR
        elif ego_longitudinal == 'd':
            ego_img += EGO_DECELERATING_COLOUR
        elif ego_longitudinal == 'c':
            ego_img += EGO_CRUISING_COLOUR
        else:
            ego_img += EGO_ISSUE
            has_issue_longitudinal = True

        # processing lateral information
        if ego_lateral == 'r':
            # we place the blinker at the front right
            x0 = EGO_IMG_W - BLINKER_W
            y0 = EGO_IMG_H - BLINKER_H
            ego_img[y0:y0 + BLINKER_H, x0:x0 + BLINKER_W] = EGO_BLINKER_COLOUR
        elif ego_lateral == 'l':
            # we place the blinker at the front left
            x0 = EGO_IMG_W - BLINKER_W
            y0 = 0
            ego_img[y0:y0 + BLINKER_H, x0:x0 + BLINKER_W] = EGO_BLINKER_COLOUR
        elif ego_lateral == 'fl':
            # lane following => no blinker
            pass
        else:
            ego_img += EGO_ISSUE
            has_issue_lateral = True

        cell_img = np.ones((CELL_IMG_H, CELL_IMG_W, 3))
        cell_img += BG_COLOUR

        # positioning ego in cell
        egoX = int((CELL_IMG_W - EGO_IMG_W) / 2)
        egoY = int((CELL_IMG_H - EGO_IMG_H) / 2)
        cell_img[egoY: egoY + EGO_IMG_H, egoX:egoX + EGO_IMG_W] = ego_img

        if debug:
            return cell_img, has_issue_lateral, has_issue_longitudinal
        else:
            return cell_img, has_issue_lateral or has_issue_longitudinal

    def make_target_img_cell(self, i, target_idx, target_lateral, target_longitudinal,
                             target_velocity, debug=False):
        if debug:
            print(i, target_idx, target_lateral, target_longitudinal, target_pos_lateral,
                  target_pos_longitudinal, target_velocity)

        cell_img = np.ones((CELL_IMG_H, CELL_IMG_W, 3))
        cell_img += BG_COLOUR

        # is there a target at all?
        if np.sum(np.array([target_lateral, target_longitudinal, target_velocity]) == '') == 3:
            # all five features are '' => no target => return empty cell
            return False, cell_img, False

        # we have an object!
        target_img = np.zeros((TARGET_IMG_H, TARGET_IMG_W, 3))
        has_issue_longitudinal = False
        has_issue_lateral = False
        has_issue_velocity = False

        # processing longitudinal information
        if target_longitudinal == 'a':
            target_img += TARGET_ACCELERATING_COLOUR
        elif target_longitudinal == 'd':
            target_img += TARGET_DECELERATING_COLOUR
        elif target_longitudinal == 'c':
            target_img += TARGET_CRUISING_COLOUR
        else:
            target_img += TARGET_ISSUE
            has_issue_longitudinal = True

        # processing lateral information
        if target_lateral in ['ro', 'ri']:
            # we place the blinker at the front right
            x0 = TARGET_IMG_W - BLINKER_W
            y0 = TARGET_IMG_H - BLINKER_H
            target_img[y0:y0 + BLINKER_H, x0:x0 + BLINKER_W] = TARGET_BLINKER_COLOUR
        elif target_lateral in ['lo', 'li']:
            # we place the blinker at the front left
            x0 = TARGET_IMG_W - BLINKER_W
            y0 = 0
            target_img[y0:y0 + BLINKER_H, x0:x0 + BLINKER_W] = TARGET_BLINKER_COLOUR
        elif target_lateral == 'fl':
            # lane following => no blinker
            pass
        else:
            target_img += TARGET_ISSUE
            has_issue_lateral = True

        # processing velocity information
        targetX = None
        targetY = int((CELL_IMG_H - TARGET_IMG_H) / 2)
        if target_velocity == 'equal':
            # same speed => centred
            targetX = int((CELL_IMG_W - TARGET_IMG_W) / 2)
        elif target_velocity == 'slower':
            # draw the "slower than" lines
            # draw "skidmarks" at the very right
            targetX = 2
            for x in range(CELL_IMG_W - TARGET_IMG_W - 2 + 1, CELL_IMG_W - 2 + 1, 2):
                for y in range(targetY, targetY + TARGET_IMG_H, 2):
                    cell_img[y:y + 1, x:x + 1] = TARGET_REL_SPEED_COLOUR
        elif target_velocity == 'faster':
            # at the very left
            targetX = CELL_IMG_W - TARGET_IMG_W - 2
            # draw the "faster than" lines
            for x in range(2, CELL_IMG_W - TARGET_IMG_W):
                for y in range(targetY, targetY + TARGET_IMG_H, 2):
                    cell_img[y:y + 1, x:x + 1] = TARGET_REL_SPEED_COLOUR
        else:
            # unknown: centre image + issue
            targetX = int((CELL_IMG_W - TARGET_IMG_W) / 2)
            has_issue_velocity = True

        # pasting target in cell
        cell_img[targetY: targetY + TARGET_IMG_H, targetX: targetX + TARGET_IMG_W] = target_img

        if debug:
            return True, cell_img, has_issue_lateral, has_issue_longitudinal, has_issue_velocity
        else:
            return True, cell_img, has_issue_lateral or has_issue_longitudinal or has_issue_velocity

    def make_macro_cell_EGO(self, i, ego_img):
        # this one is simple
        macro_cell_img = np.zeros((MACRO_CELL_H, MACRO_CELL_W, 3))
        macro_cell_img += BG_COLOUR

        egoX = int((MACRO_CELL_W - CELL_IMG_W) / 2)
        egoY = int((MACRO_CELL_H - CELL_IMG_H) / 2)

        macro_cell_img[egoY:egoY + CELL_IMG_H, egoX:egoX + CELL_IMG_W] = ego_img

        return macro_cell_img

    def make_macro_cell_target(self, i, lateral_position, longitudinal_position, target_img_list,
                               target_position):

        macro_cell_img = np.zeros((MACRO_CELL_H, MACRO_CELL_W, 3))
        macro_cell_img += BG_COLOUR

        for t in range(n_targets):
            target_img = target_img_list[t]
            target_pos = target_position[t]

            if target_pos['lateral'] == lateral_position and target_pos[
                'longitudinal'] == longitudinal_position:
                # this target can be added to the macro cell
                # print(t)
                targetX = int(int(t / 4) * MACRO_CELL_W / 2)
                targetY = int(t % 4 * MACRO_CELL_H / 4)
                macro_cell_img[targetY: targetY + CELL_IMG_H,
                targetX: targetX + CELL_IMG_W] = target_img

        return macro_cell_img

    def make_full_img(self, i, ego_img, target_img_list, target_rel_pos):

        # initialising full image
        n_rows = len(
            valid_relative_positions['lateral']) - 1  # -1 because 'lateral' also contains ''
        n_cols = len(valid_relative_positions['longitudinal'])  # so basically all longitudinal + EGO CELL which it is always at centre
        full_img = np.zeros((n_rows * MACRO_CELL_H + (n_rows - 1) * H_LANE_H,
                             n_cols * MACRO_CELL_W + (n_cols - 1) * V_LANE_W, 3))
        full_img += BG_COLOUR

        # print(n_rows, n_cols)

        # adding horizontal lanes
        for i in range(1, n_rows):
            y0 = MACRO_CELL_H * i + (i - 1) * H_LANE_H
            for x in range(0, full_img.shape[1], 2 * H_LANE_W):
                full_img[y0: y0 + H_LANE_H, x: x + H_LANE_W] = H_LANE_COLOUR

        # adding vertical lanes
        for j in range(1, n_cols):
            x0 = MACRO_CELL_W * j + (j - 1) * V_LANE_W
            for y in range(0, full_img.shape[0], 2 * V_LANE_H):
                full_img[y: y + V_LANE_H, x0: x0 + V_LANE_W] = V_LANE_COLOUR

        # ego img positioning: always in the centre
        ego_col = int(n_cols / 2)
        x0 = ego_col * MACRO_CELL_W + int((n_cols - 1) / 2) * V_LANE_W
        y0 = int(n_rows / 2) * MACRO_CELL_H + int((n_rows - 1) / 2) * H_LANE_H
        macro_cell_ego = self.make_macro_cell_EGO(i, ego_img)
        full_img[y0: y0 + MACRO_CELL_H, x0: x0 + MACRO_CELL_W] = macro_cell_ego

        for lat_i in range(len(valid_relative_positions['lateral'])):
            lat = valid_relative_positions['lateral'][lat_i]
            if lat == '':
                continue
            y0 = MACRO_CELL_H * lat_i
            if lat_i > 0:
                y0 += lat_i * H_LANE_H

            x0 = None
            for lon_i in range(len(valid_relative_positions['longitudinal'])):
                lon = valid_relative_positions['longitudinal'][lon_i]
                if lon == '':
                    continue
                x0 = lon_i * MACRO_CELL_W
                if lon_i > 0:
                    x0 += lon_i * V_LANE_W
                if lon_i >= ego_col:
                    x0 += MACRO_CELL_W + V_LANE_W

                macro_cell_target = self.make_macro_cell_target(i, lat, lon, target_img_list,
                                                                target_rel_pos)
                full_img[y0: y0 + MACRO_CELL_H, x0: x0 + MACRO_CELL_W] = macro_cell_target

        return full_img

    def initialise_dictionary(self, print_debug):
        # either load existing one or create an empty one

        if print_debug:
            print("\n** INITIALISING DICTIONARY **")

        if self.dictionary_filename in os.listdir(self.input_path_dictionary):
            if print_debug:
                print("\tloading", self.output_path_dictionary + self.dictionary_filename, end=' ')
            self.dictionary = pd.read_pickle(self.output_path_dictionary + self.dictionary_filename)
        else:
            if print_debug:
                print("\tcreating new dictionary, n_targets =", n_targets, end=' ')

            self.dictionary = pd.DataFrame(columns=dictionary_columns)

        if print_debug:
            print(self.dictionary.shape)

    def file_and_folder_preparation(self, print_debug):

        # folder settings etc
        data_folder = '..\\data\\'
        base_folder = data_folder + '\\'
        input_folder = "00_raw_data"
        output_folder = "01_activity_dictionary"
        img_outputSubFolder = "img_" + str(
            len(valid_relative_positions['lateral']) - 1) + "x" + str(
            len(valid_relative_positions['longitudinal']) - 1)

        self.dictionary_filename = "activityDictionary_" + str(
            len(valid_relative_positions['lateral']) - 1) + "x" + str(
            len(valid_relative_positions['longitudinal']) - 1) + ".csv"
        self.input_path = base_folder + input_folder + "\\"
        self.input_path_dictionary = base_folder + output_folder + "\\"
        self.output_path_dictionary = base_folder + output_folder + "\\"
        self.output_path_dictionary_img = self.output_path_dictionary + img_outputSubFolder + "\\"

        if print_debug:
            print("** FILE AND FOLDER SETTINGS **")
            print("\tbase_folder:", base_folder)
            print("\tinput_folder:", input_folder)
            print("\toutput_folder:", output_folder)
            print("\timg_outputSubFolder:", img_outputSubFolder)
            print("\tdictionary_filename:", self.dictionary_filename)
            print("\tinput_path:", self.input_path)
            print("\tinput_path_dictionary:", self.input_path_dictionary)
            print("\toutput_path_dictionary:", self.output_path_dictionary)
            print("\toutput_path_dictionary_img:", self.output_path_dictionary_img)

        if print_debug:
            print("\n** FOLDER INITIALISATIONS **")

        if output_folder not in os.listdir(base_folder):
            if print_debug:
                print("\tcreating", output_folder)
            os.mkdir(self.output_path_dictionary)
        else:
            if print_debug:
                print("\tpath", self.output_path_dictionary, "already exists")

        if img_outputSubFolder not in os.listdir(self.output_path_dictionary):
            if print_debug:
                print("\tcreating", self.output_path_dictionary_img)
            os.mkdir(self.output_path_dictionary_img)
        else:
            if print_debug:
                print("\tpath", self.output_path_dictionary_img, "already exists")

    def load_dataset(self, input_file, print_debug):

        # # DATASET LOADING
        if print_debug:
            print("\t01. loading", input_file, "..", end='')
        s = pd.HDFStore(self.input_path + input_file)
        tagged_dataset = s.get('df')
        s.close()
        tagged_dataset.reset_index(inplace=True, drop=True)

        if print_debug:
            print("OK, shape =", tagged_dataset.shape)

        # # TAGGED DATAFRAME CORRECTION BASED ON VALID RELATIVE POSITIONS
        if print_debug:
            print("\t02. dataset correction based on dictionary grid-size..", end='')
        # print(dictionary.shape, len(core_dictionary_columns), len(img_dictionary_columns))
        # print(tagged_dataset[core_dictionary_columns].shape)
        # print(tagged_dataset[core_dictionary_columns].drop_duplicates().shape)

        for t in range(n_targets):
            tagged_dataset.loc[(~tagged_dataset['target_' + str(t) + '_relative_position_longitudinal'].isin(valid_relative_positions['longitudinal'])) |
                               (~tagged_dataset['target_' + str(t) + '_relative_position_lateral'].isin(valid_relative_positions['lateral'])),
                               ['target_' + str(t) + '_lateral',
                                'target_' + str(t) + '_longitudinal',
                                'target_' + str(t) + '_relative_position_longitudinal',
                                'target_' + str(t) + '_relative_position_lateral',
                                'target_' + str(t) + '_velocity']] = ''

        if print_debug:
            print("OK")

        return tagged_dataset

    def save_dictionary(self, print_debug):
        if print_debug:
            print("\t04. finalisation..")

        files = os.listdir(self.output_path_dictionary_img)

        for i in range(len(self.dictionary)):

            if print_debug:
                print('\t\t', i + 1, '/', len(self.dictionary), 10 * ' ', sep='', end='\r')

            # we should check whether the image already exists
            if str(i) + ".jpg" in files:
                continue

            # render(self.dictionary.iloc[i]['full_img'], saveFig=True, title=str(i),
            #        outPath=self.output_path_dictionary_img)

        # drop all columns not part of dictionary_columns (i.e. full_img)
        remove_cols = [c for c in self.dictionary.columns if c not in dictionary_columns]
        self.dictionary.drop(columns=remove_cols, inplace=True)

        self.dictionary.to_pickle(self.output_path_dictionary + self.dictionary_filename)

        if print_debug:
            print('\n\t\tOK')

    def dictionary_update(self, input_file, print_debug):

        if print_debug:
            print("\n** DICTIONARY UPDATE **")

        reduced_dataset = self.load_dataset(input_file, print_debug)

        # # DICTIONARY CONSTRUCTION OF THE CORRECTED DATAFRAME
        # so basically we check which entries are new wrt the existing dictionary
        if print_debug:
            print("\t03. dictionary construction of the corrected dataframe (shape=",
                  reduced_dataset[core_dictionary_columns].drop_duplicates().shape, "..", end='')

        unique_entries = reduced_dataset[core_dictionary_columns].drop_duplicates()
        unique_entries.reset_index(inplace=True, drop=True)

        # DICTIONARY ENTRY CONSTRUCTION
        n_add = 0
        n_skip = 0

        if print_debug:
            print("    ", end='')

        for i in range(len(unique_entries)):
            if print_debug:
                print("*" if (i + 1) % 10 == 0 else ".", end='')

            # is the entry already present in the dictionary?
            entry = unique_entries.iloc[i]

            add_entry = False
            if len(self.dictionary) == 0:
                add_entry = True
            elif np.max(np.sum(self.dictionary[entry.index] == entry, axis=1)) < len(entry):
                # the highest number of common features is less than the lenght of the entry => new feature values => add!
                add_entry = True

            if add_entry:
                # we can add this entry
                entry['index'] = len(self.dictionary)
                entry['n_objects'] = sum(
                    [entry['target_' + str(t) + "_lateral"] != '' for t in range(n_targets)])
                entry['ego_img'], entry['ego_issue'] = self.make_EGO_IMG_cell(i, entry['host_lateral'], entry['host_longitudinal'])

                target_position = []
                for t in range(n_targets):
                    target_lateral = entry['target_' + str(t) + '_lateral']
                    target_longitudinal = entry['target_' + str(t) + '_longitudinal']
                    target_velocity = entry['target_' + str(t) + '_velocity']
                    _, entry['target_'+str(t)+'_img'], entry['target_'+str(t)+'_issue'] = \
                        self.make_target_img_cell(i, t, target_lateral, target_longitudinal, target_velocity)
                    target_position.append(
                        {'lateral': entry['target_' + str(t) + '_relative_position_lateral'],
                         'longitudinal': entry['target_' + str(t) + '_relative_position_longitudinal']})

                # time to merge the EGO/target images into a macro image
                # entry['full_img'] = self.make_full_img(i, entry['ego_img'],
                #                                        [entry['target_' + str(x) + '_img'] for x in
                #                                         range(n_targets)], target_position)

                # sanity check: do we have all entries to add to the dictionarY?
                all_good = sum([c in self.dictionary.columns for c in entry.index]) == len(self.dictionary.columns)
                all_good = True
                if all_good:
                    # print("all good to add!")
                    n_add += 1

                    # print(self.dictionary.columns)
                    # print(list(entry.index))

                    # self.dictionary = self.dictionary.append(entry, ignore_index=True)
                    entry_dict = {k: entry[k] for k in self.dictionary.columns}
                    # entry_dict['full_img'] = entry['full_img']
                    # print(entry_dict)
                    self.dictionary = self.dictionary.append(entry_dict, ignore_index=True)

            else:
                # print("already present")
                n_skip += 1

        # break

        if print_debug:
            print("\n\t\tdataset processed, new entries =", n_add, "existing entries =", n_skip)
            print("\t\tupdated dictionary shape=", self.dictionary.shape)
            print("\t\tissue status:")
            for c in self.dictionary.columns:
                if 'issue' in c:
                    print("\t\t", c, sum(self.dictionary[c]))

    def run(self, file_to_parse, print_debug):

        if file_to_parse == 'all':
            file_list = os.listdir(self.input_path)

            for f in file_list:
                self.dictionary_update(f, print_debug)
        else:
            self.dictionary_update(file_to_parse, print_debug)

        self.save_dictionary(print_debug)

        if print_debug:
            print("\n** END OF DICTIONARY UPDATE**")
            print("\tyou can find the dictionary here:")
            print("\t\t", self.output_path_dictionary, self.dictionary_filename)
            print("\tthe dictionary's images are here:")
            print("\t\t", self.output_path_dictionary_img)
            print("\nBye!")


if __name__ == "__main__":

    arg_list = [{'arg_name': 'file_to_parse', 'descr': "either filename or 'all'"},
                {'arg_name': 'print_debug', 'descr': '0 for false, 1 for true'}]

    if len(sys.argv[1:]) != len(arg_list):
        print("arg list:")
        for i in range(len(arg_list)):
            print(i, 4 * ' ', arg_list[i]['arg_name'], ':', arg_list[i]['descr'])

    else:
        arguments = ['script_name'] + [x['arg_name'] for x in arg_list]

        filename = sys.argv[arguments.index('file_to_parse')]
        print_debug = int(sys.argv[arguments.index('print_debug')])

        DU = dictionary_update(print_debug)
        DU.run(filename, print_debug)
