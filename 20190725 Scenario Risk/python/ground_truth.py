""" Calculating the performance of the scenario mining for the IV paper.

Creation date: 2020 01 12
Author(s): Erwin de Gelder

Modifications:
"""

import os
import numpy as np
from activity_detector import LateralActivityHost, LeadVehicle, LateralActivityTarget
from ngram import NGram
from ngram_search import find_sequence

GROUND_TRUTH = {"20170524_PP_01_Run_1": dict(cutin=[1799.29, 1960.68, 1982.01, 2370.28, 2465.04,
                                                    2558.85, 2677.92],
                                             overtaking=[1814.72, 2546.03]),
                "20170524_PP_01_Run_2": dict(cutin=[1612.83, 1615.57, 2015.97, 2088.14, 2276.01,
                                                    2458.68, 2484.91, 2486.17]),
                "20170524_PP_01_Run_3": dict(cutin=[2282.34, 2554.09, 2567.49, 2575.77, 2608.50]),
                "20170524_PP_02_Run_1": dict(cutin=[277.20, 1779.44, 1841.29, 1851.44, 1912.88,
                                                    1980.18, 2391.17, 2430.47, 2432.25, 2531.54,
                                                    2680.97])}


def empty_ngram_models():
    """ Return the empty n-gram models of the targets and ego vehicle. """
    fieldnames = ["longitudinal_activity", "lateral_activity", "longitudinal_state",
                  "lateral_state", "lead_vehicle", "id"]
    metadata = (("tstart", float), ("tend", float), ("target_id", int))
    target_ngrams = NGram(fieldnames, metadata)
    fieldnames = ["host_longitudinal_activity", "host_lateral_activity", "is_highway"]
    metadata = (("tstart", float), ("tend", float))
    ego_ngram = NGram(fieldnames, metadata)
    return target_ngrams, ego_ngram


def compute_performance(name, target_tags, ego_tags):
    """ Compute the performance. """
    verbose = False
    result = []  # [[FN, FP, TP]]
    print("{:20s} FN FP TP".format(""))
    for key, item in GROUND_TRUTH.items():
        if name in item:
            # Load the n-gram models.
            filename = os.path.join("data", "4_ngrams", "{:s}.hdf5".format(key))
            targets, ego = empty_ngram_models()
            if not targets.from_hdf(filename, "targets"):
                raise KeyError("The n-gram models for the targets are not defined.")
            if not ego.from_hdf(filename, "ego"):
                raise KeyError("The n-gram model for the ego vehicle is not defined.")

            # Get the matches.
            matches = []
            for i, target in enumerate(targets.ngrams):
                search = find_sequence((target, ego.ngram), (target_tags, ego_tags))
                while search.is_found:
                    if verbose:
                        print(i, search)
                    matches.append(search)
                    search = find_sequence((target, ego.ngram), (target_tags, ego_tags),
                                           t_start=search.t_end+5)

            # Loop through the ground truth and check for false negatives.
            fnfptp = [0, 0, 0]
            for index in item[name]:
                false_negative = True
                for match in matches:
                    if match.t_start <= index <= match.t_end:
                        false_negative = False
                        break
                if false_negative:
                    fnfptp[0] += 1

            # Loop through the matches and check for false positives and true positives.
            for match in matches:
                true_positive = False
                for index in item[name]:
                    if match.t_start <= index <= match.t_end:
                        true_positive = True
                        break
                if true_positive:
                    fnfptp[2] += 1
                else:
                    fnfptp[1] += 1
            print("{:20s} {:2d} {:2d} {:2d}".format(key, fnfptp[0], fnfptp[1], fnfptp[2]))
            result.append(fnfptp)

    result = np.sum(np.array(result), axis=0)
    print("Recall: {:.1f} %".format(result[2] / (result[0] + result[2]) * 100))
    print("Precision: {:.1f} %".format(result[2] / (result[1] + result[2]) * 100))


# Check for cut-ins.
TARGET_TAGS = [dict(lateral_activity=[LateralActivityTarget.LEFT_CUT_IN.value,
                                      LateralActivityTarget.RIGHT_CUT_IN.value],
                    lead_vehicle=[LeadVehicle.NOLEAD.value]),
               dict(lateral_activity=[LateralActivityTarget.LEFT_CUT_IN.value,
                                      LateralActivityTarget.RIGHT_CUT_IN.value],
                    lead_vehicle=[LeadVehicle.LEAD.value])]
EGO_TAGS = [dict(host_lateral_activity=[LateralActivityHost.LANE_FOLLOWING.value],
                 is_highway=[True]),
            dict(host_lateral_activity=[LateralActivityHost.LANE_FOLLOWING.value],
                 is_highway=[True])]
compute_performance("cutin", TARGET_TAGS, EGO_TAGS)
