""" Use the ERP2017 data to create the ngrams

Creation date: 2020 03 13
Author(s): Erwin de Gelder

Modifications:
2020 04 26 Add tag of near targets to n-gram models.
"""


import argparse
from glob import glob
import os
import multiprocessing as mp
from data_handler import DataHandler
from ngram import NGram


PARSER = argparse.ArgumentParser(description="Create n-grams")
PARSER.add_argument('-complevel', default=4, type=int, choices=range(10),
                    help="Compression level, default=4")
PARSER.add_argument('-inputfolder', default=os.path.join("data", "1_hdf5"), type=str,
                    help="Folder to read data from")
PARSER.add_argument('-outputfolder', default=os.path.join("data", "4_ngrams"), type=str,
                    help="Folder to write new data to")
PARSER.add_argument('-file', default=None, type=str, help="If not all files, select single file")
ARGS = PARSER.parse_args()


FIELDNAMES_TARGET = ["longitudinal_activity", "lateral_activity", "longitudinal_state",
                     "lateral_state", "lead_vehicle", "near", "id"]
METADATA_TARGET = (("tstart", float), ("tend", float), ("target_id", int))
FIELDNAMES_EGO = ["host_longitudinal_activity", "host_lateral_activity", "is_highway"]
METADATA_EGO = (("tstart", float), ("tend", float))


def process_file(filename_input: str, filename_output: str) -> None:
    """ Create an n-gram from a single HDF5 file.

    :param filename_input: Path of the to-be-processed file.
    :param filename_output: Path of to-be-created file with the n-grams.
    """
    try:
        # Load the data
        data_handler = DataHandler(filename_input)

        # Process the targets
        target_ngrams = NGram(FIELDNAMES_TARGET, METADATA_TARGET)
        for target in data_handler.targets:
            target_ngrams.ngram_from_data(target,
                                          tstart=target.index[0],
                                          tend=target.index[-1],
                                          target_id=int(target["id"].values[0]))
        target_ngrams.sort_ngrams("tstart")
        target_ngrams.to_hdf(filename_output, "targets", mode="w")

        # Process the ego vehicle.
        ego_ngram = NGram(FIELDNAMES_EGO, METADATA_EGO)
        ego_ngram.ngram_from_data(data_handler.data,
                                  tstart=data_handler.data.index[0],
                                  tend=data_handler.data.index[-1])
        ego_ngram.to_hdf(filename_output, "ego")

    except Exception as exception:
        print("Error at file: {:s}".format(filename_input))
        raise exception


if __name__ == "__main__":
    if not os.path.exists(ARGS.inputfolder):
        print("Provided folder '{:s}' does not exist.".format(ARGS.folder))
        exit()
    if not os.path.exists(ARGS.outputfolder):
        os.mkdir(ARGS.outputfolder)
    if ARGS.file is None:
        DATAFILES = glob(os.path.join(ARGS.inputfolder, '*.hdf5'))
        ARGUMENTS = [(input_file, os.path.join(ARGS.outputfolder, os.path.basename(input_file)))
                     for input_file in DATAFILES]
        with mp.Pool(processes=4) as POOL:
            for _ in POOL.starmap(process_file, ARGUMENTS):
                pass
    else:
        if not os.path.exists(os.path.join(ARGS.inputfolder, ARGS.file)):
            print("Provided file '{:s}' does not exist.".format(os.path.join(ARGS.inputfolder,
                                                                             ARGS.file)))
            exit()
        process_file(os.path.join(ARGS.inputfolder, ARGS.file),
                     os.path.join(ARGS.outputfolder, ARGS.file))
