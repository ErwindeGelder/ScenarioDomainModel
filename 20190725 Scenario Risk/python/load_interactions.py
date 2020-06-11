""" Functionality for loading the NGSIM interactions.

Creation date: 2020 06 10
Author(s): Jeroen Manders, Erwin de Gelder

Modifications:
"""

# Standard Library Imports
from glob import glob
import os


# Related Third Party Imports
import pandas as pd
from tqdm import tqdm


# Local Application/Library Specific Imports


def load_interaction(file_path):
    """ Loads a single interaction hdf5.

    Args:
        file_path: path of interaction hdf5 file

    Returns:
        interaction: dict of interaction

    """
    with pd.HDFStore(file_path, mode="r") as hdf_store:
        interaction_dict = dict(id=os.path.splitext(os.path.basename(file_path))[0],
                                vehicle_rear=hdf_store.get("vehicle_rear"),
                                vehicle_front=hdf_store.get("vehicle_front"),
                                interaction=hdf_store.get("interaction"))
    return interaction_dict


if __name__ == "__main__":
    NUM_INTERACTIONS = 100
    # Load first NUM_INTERACTIONS interactions
    interactions = dict()
    for interaction_path in tqdm(glob("interactions/*.hdf5")[:NUM_INTERACTIONS]):
        interaction = load_interaction(interaction_path)
        interactions[interaction["id"]] = interaction
