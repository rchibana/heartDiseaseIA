__author__ = 'rchibana'

import pandas as pd

DATASET_PATH = "dataset/hungarian.csv"
DELIMITER = ","


def encode_target(data_frame, target_column):
    """Add column to data_frame (dataframe) with integers for the target.

    Args
    ----
    data_frame -- pandas DataFrame.
    target_column -- column to map to int, producing new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """

    df_mod = data_frame.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return df_mod, targets


def get_data():
    """Get the data from a file"""

    data_frame = pd.read_csv(DATASET_PATH, delimiter=DELIMITER)
    return data_frame
