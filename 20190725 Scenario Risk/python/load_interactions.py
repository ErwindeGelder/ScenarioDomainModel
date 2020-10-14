""" Functionality for loading the NGSIM interactions.

Creation date: 2020 06 10
Author(s): Jeroen Manders, Erwin de Gelder

Modifications:
2020 10 13: Add functionality to save interaction.
"""

# Standard Library Imports
from glob import glob
import os


# Related Third Party Imports
import pandas as pd
from tqdm import tqdm


# Local Application/Library Specific Imports


def load_interaction(file_path: str) -> dict:
    """ Loads a single interaction hdf5.

    :param file_path: path of interaction hdf5 file.
    :Returns: dict of interaction.
    """
    with pd.HDFStore(file_path, mode="r") as hdf_store:
        interaction_dict = dict(id=os.path.splitext(os.path.basename(file_path))[0],
                                vehicle_rear=hdf_store.get("vehicle_rear"),
                                vehicle_front=hdf_store.get("vehicle_front"),
                                interaction=hdf_store.get("interaction"))
    return interaction_dict


def save_interaction(interaction_dict: dict, folder: str = os.path.join("data", "8_interactions")) \
        -> None:
    """ Store the interaction data in an HDF5 file.

    :param interaction_dict: The dictionary with the interaction data.
    :param folder: Folder to store the file in (default: data/8_interactions).
    """
    filename = os.path.join(folder, "{:s}.hdf5".format(interaction_dict["id"]))
    with pd.HDFStore(filename, mode="w") as hdf_store:
        for key in ["vehicle_rear", "vehicle_front", "interaction"]:
            hdf_store.put(key, interaction_dict[key], format='table')


if __name__ == "__main__":
    NUM_INTERACTIONS = 100
    # Load first NUM_INTERACTIONS interactions
    interactions = dict()
    for interaction_path in tqdm(glob("interactions/*.hdf5")[:NUM_INTERACTIONS]):
        interaction = load_interaction(interaction_path)
        interactions[interaction["id"]] = interaction
