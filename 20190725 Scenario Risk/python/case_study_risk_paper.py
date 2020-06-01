""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
"""

import os
import json
import numpy as np
from domain_model import Scenario
from stats import KDE
from databaseemulator import DataBaseEmulator


OVERWRITE = False  # If true, all results will be obtained again.


def parameters(scenario: Scenario) -> np.ndarray:
    """ Get the parameters of the ego-braking scenario.

    It is assumed that the secnario contains only one scenario.
    This activity describes the braking activity of the ego
    vehicle. The following parameters are obtained:
    - Initial speed
    - Average deceleration
    - Speed difference

    :param scenario: The scenario describing the braking activity of the ego
        vehicle.
    :return: The mentioned parameters.
    """
    activity = scenario.activities[0]
    vstart_vend = activity.get_state(time=[activity.tstart, activity.tend])
    vstart = vstart_vend[0]
    vdiff = (vstart - vstart_vend[1])
    amean = vdiff / (activity.tend - activity.tstart)
    return np.array([vstart, amean, vdiff])


def kde_lead_braking() -> KDE:
    """ Return the KDE for the braking parameters.

    For now, the scenarios with the ego vehicle braking are used. This should
    later be changed to scenarios with the lead vehicle braking.

    :return: KDE of the braking parameters.
    """
    filename_kde = os.path.join("data", "6_kde", "ego_braking.json")
    if os.path.exists(filename_kde) and not OVERWRITE:
        with open(filename_kde, "r") as file:
            save = json.load(file)
        kde = KDE(np.array(save["data"]), scaling=True)
        kde.data_helpers.std = np.array(save["std"])  # Data is already scaled, so provide std.
        kde.bandwidth = save["bandwidth"]
    else:
        # Create the KDE.
        filename_data = os.path.join("data", "5_scenarios", "ego_braking.json")
        dbe = DataBaseEmulator(filename_data)
        nscenarios = len(dbe.collections["scenario"])
        pars = np.array([parameters(dbe.get_item("scenario", i)) for i in range(nscenarios)])
        kde = KDE(pars, scaling=True)
        kde.compute_bandwidth()
        print("KDE bandwidth ego braking: {:.3f}".format(kde.bandwidth))

        # Write the KDE through a JSON file.
        save = dict(data=kde.data.tolist(),
                    std=kde.data_helpers.std.tolist(),
                    bandwidth=kde.bandwidth)
        if not os.path.exists(os.path.dirname(filename_kde)):
            os.mkdir(os.path.dirname(filename_kde))
        with open(filename_kde, "w") as file:
            json.dump(save, file)

    return kde


if __name__ == "__main__":
    kde = kde_lead_braking()
    print(kde.data_helpers.std)
    print(kde.sample(10))
