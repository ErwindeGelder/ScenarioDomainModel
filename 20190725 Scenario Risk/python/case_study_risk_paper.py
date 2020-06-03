""" Do the simulations for the case study of the scenario risk paper.

Creation date: 2020 05 31
Author(s): Erwin de Gelder

Modifications:
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from domain_model import Scenario
from simulation import SimulationLeadBraking
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


def check_validity_lead_braking(par: np.ndarray) -> bool:
    """ Check whether the parameters are valid.

    :param par: A vector of the parameters.
    :return: Whether the parameters are valid.
    """
    # Speed should be decreasing.
    if par[2] <= 0:
        return False

    # End speed should not be negative.
    if par[2] > par[0]:
        return False

    # Mean deceleration should be positive.
    if par[1] <= 0:
        return False

    return True


def simulation_lead_braking(nsim: int = 1000) -> pd.DataFrame:
    """ Do the initial simulations for the lead braking scenario.

    :param nsim: Number of simulations.
    :return: The resulting probabilities of failures (P(B)).
    """
    filename_df = os.path.join("data", "7_simulation_results", "lead_braking.csv")

    if os.path.exists(filename_df) and not OVERWRITE:
        df = pd.read_csv(filename_df, index_col=0)
    else:
        # Obtain the KDE for sampling and generate the parameters.
        kde = kde_lead_braking()
        np.random.seed(0)
        pars = np.zeros((nsim, 3))
        i = 0
        while i < nsim:
            pars[i, :] = kde.sample()
            if check_validity_lead_braking(pars[i, :]):
                i += 1
        df = pd.DataFrame(data=pars, columns=["v0", "amean", "dv"])

        # Loop through each parameter vector and obtain the simulation result.
        simulator = SimulationLeadBraking()
        result = np.zeros(nsim)
        for i, par in enumerate(tqdm(pars)):
            result[i] = simulator.get_probability(tuple(par))
        df["result"] = result

        # Write to file.
        # Write the KDE through a JSON file.
        if not os.path.exists(os.path.dirname(filename_df)):
            os.mkdir(os.path.dirname(filename_df))
        df.to_csv(filename_df)

    return df


if __name__ == "__main__":
    print(simulation_lead_braking())
