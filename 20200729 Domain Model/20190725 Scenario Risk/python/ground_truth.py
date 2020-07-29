""" Calculating the performance of the scenario mining for the IV paper.

Creation date: 2020 01 12
Author(s): Erwin de Gelder

Modifications:
"""

import os
import numpy as np
from activity_detector import LateralActivityHost, LeadVehicle, LateralActivityTarget, \
    LongitudinalStateTarget, LateralStateTarget
from ngram import NGram
from ngram_search import find_sequence

GROUND_TRUTH = {"20170524_PP_01_Run_1": dict(cutin=[1799.29, 1960.68, 1982.01, 2370.28, 2465.04,
                                                    2558.85, 2677.92],
                                             overtaking=[1810., 2546.01],
                                             host_llc=[203.35, 217.29, 1738.83, 1813.3, 1900.71,
                                                       1963.60, 2000.3, 2171.98, 2310.65, 2318.23,
                                                       2546.01, 2650.02],
                                             host_rlc=[251.46, 304.95, 1846.39, 1858.69, 1927.05,
                                                       2040.03, 2193.44, 2236.72, 2487.13, 2568.65,
                                                       2668.15, 2675.00]),
                "20170524_PP_01_Run_2": dict(cutin=[1612.83, 1615.57, 2015.97, 2088.14, 2276.01,
                                                    2458.68, 2484.91, 2486.17],
                                             overtaking=[1559.90, 1739.87],
                                             host_llc=[187.75, 1559.80, 1739.36, 1742.11, 1799.,
                                                       1839.25, 2009.09, 2144.10, 2152.10],
                                             host_rlc=[288.64, 1692.84, 1758.40, 1888., 2029.03,
                                                       2075.3, 2498.02, 2522.54]),
                "20170524_PP_01_Run_3": dict(cutin=[2282.34, 2554.09, 2567.49, 2575.77, 2608.50],
                                             overtaking=[2349.51, 2471., 2483.52],
                                             host_llc=[177.34, 1697.09, 1922.35, 1957.28, 2191.60,
                                                       2324.75, 2352.08, 2485.81],
                                             host_rlc=[274.34, 1882.64, 2000.59, 2212.74, 2256.5,
                                                       2400.38, 2619.93, 2699.48]),
                "20170524_PP_02_Run_1": dict(cutin=[277.20, 1779.44, 1841.29, 1851.44, 1912.88,
                                                    1980.18, 2391.17, 2430.47, 2432.25, 2531.54,
                                                    2680.97],
                                             overtaking=[226.84, 226.84, 247.56, 250.19, 2312.99,
                                                         2410.64, 2573.86],
                                             host_llc=[208.43, 226.84, 247.56, 1727.20, 1780.99,
                                                       1958.41, 2013.89, 2217.42, 2312.99, 2359.64,
                                                       2410.64, 2573.86, 2692.67],
                                             host_rlc=[269.45, 280.92, 290.23, 1844.99, 1910.55,
                                                       2057.85, 2242.24, 2326.85, 2598.55, 2602.75,
                                                       2709.19, 2714.15]),
                "20170524_PP_02_Run_2": dict(cutin=[1609.68, 2348.73, 2470.87, 2574.56, 2645.88],
                                             overtaking=[232.38, 232.38, 2401.47, 2421.25, 2622.38],
                                             host_llc=[203.37, 232.38, 1607.12, 1865.63, 2007.19,
                                                       2197.54, 2329.22, 2401.47, 2421.25,
                                                       2534.43, 2569.20, 2622.38, 2657.46],
                                             host_rlc=[259.02, 290.28, 1826.11, 2082.7, 2221.62,
                                                       2265.1, 2430.51, 2465.65, 2551.17, 2654.54,
                                                       2665.21, 2667.37, 2690.52]),
                "20170524_PP_02_Run_3": dict(cutin=[257.36, 1846.44, 1983.09, 2007.43, 2606.19,
                                                    2629.41],
                                             overtaking=[2742.55, 2869.04],
                                             host_llc=[197.25, 1674.19, 2040.49, 2144., 2331.22,
                                                       2559.96, 2697.29, 2742.55, 2869.04],
                                             host_rlc=[282.10, 2007.43, 2440.86, 2582.6, 2626.5,
                                                       2809.70, 2890.34]),
                "20170530_PP_06_Run_1": dict(cutin=[304.23, 330.44, 2176.94, 2456.44, 2537.23],
                                             overtaking=[1941.88, 2451.34, 2506.81, 2506.81],
                                             host_llc=[267.50, 283.92, 1679.90, 1685.69, 1743.05,
                                                       1869.64, 1911.29, 1941.88, 2084.13, 2215.95,
                                                       2251.95, 2451.34, 2506.81],
                                             host_rlc=[343.79, 347.26, 1699.56, 1773.59, 1834.36,
                                                       1945.94, 1948.87, 2113.51, 2147., 2469.31,
                                                       2549.06, 2573.])
                }


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
    print()
    print("Check result for {:s}".format(name))
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
            if target_tags is not None:
                for i, target in enumerate(targets.ngrams):
                    search = find_sequence((target, ego.ngram), (target_tags, ego_tags))
                    while search.is_found:
                        if verbose:
                            print(i, search)
                        matches.append(search)
                        search = find_sequence((target, ego.ngram), (target_tags, ego_tags),
                                               t_start=search.t_end+5)
            else:
                search = find_sequence((ego.ngram,), (ego_tags,))
                while search.is_found:
                    if verbose:
                        print(search)
                    matches.append(search)
                    search = find_sequence((ego.ngram,), (ego_tags,), t_start=search.t_end)

            # Loop through the ground truth and check for false negatives and true positives.
            fnfptp = [0, 0, 0]
            for index in item[name]:
                false_negative = True
                for match in matches:
                    if match.t_start <= index <= match.t_end:
                        false_negative = False
                        break
                if false_negative:
                    fnfptp[0] += 1
                else:
                    fnfptp[2] += 1

            # Loop through the matches and check for false positives.
            for match in matches:
                true_positive = False
                for index in item[name]:
                    if match.t_start <= index <= match.t_end:
                        true_positive = True
                        break
                if not true_positive:
                    fnfptp[1] += 1
            print("{:20s} {:2d} {:2d} {:2d}".format(key, fnfptp[0], fnfptp[1], fnfptp[2]))
            result.append(fnfptp)

    result = np.sum(np.array(result), axis=0)
    recall = result[2] / (result[0] + result[2])
    precision = result[2] / (result[1] + result[2])
    print("{:20s} {:2d} {:2d} {:2d}".format("Total", result[0], result[1], result[2]))
    print("Recall: {:.1f} %".format(recall * 100))
    print("Precision: {:.1f} %".format(precision * 100))
    print("F1 score: {:.1f} %".format(200*precision*recall/(precision+recall)))


# Check for host left/right lane change.
EGO_TAGS = [dict(host_lateral_activity=[LateralActivityHost.LEFT_LANE_CHANGE.value],
                 is_highway=[True])]
compute_performance("host_llc", None, EGO_TAGS)
EGO_TAGS = [dict(host_lateral_activity=[LateralActivityHost.RIGHT_LANE_CHANGE.value],
                 is_highway=[True])]
compute_performance("host_rlc", None, EGO_TAGS)


# Check for cut-ins.
TARGET_TAGS = [dict(lateral_state=[LateralStateTarget.LEFT.value],
                    longitudinal_state=[LongitudinalStateTarget.REAR.value]),
               dict(lateral_state=[LateralStateTarget.LEFT.value],
                    longitudinal_state=[LongitudinalStateTarget.FRONT.value]),
               dict(lateral_state=[LateralStateTarget.LEFT.value],
                    longitudinal_state=[LongitudinalStateTarget.FRONT.value]),
               dict(lateral_state=[LateralStateTarget.SAME.value],
                    longitudinal_state=[LongitudinalStateTarget.FRONT.value])]
EGO_TAGS = [dict(host_lateral_activity=[LateralActivityHost.LANE_FOLLOWING.value],
                 is_highway=[True]),
            dict(host_lateral_activity=[LateralActivityHost.LANE_FOLLOWING.value],
                 is_highway=[True]),
            dict(host_lateral_activity=[LateralActivityHost.LEFT_LANE_CHANGE.value],
                 is_highway=[True]),
            dict(host_lateral_activity=[LateralActivityHost.LEFT_LANE_CHANGE.value],
                 is_highway=[True])]
compute_performance("overtaking", TARGET_TAGS, EGO_TAGS)

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
