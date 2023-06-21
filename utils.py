"""
The utils module is a collection of functions that perform small tasks.
By placing them in this module they can be reused and make other code parts more readable.
"""

import torch
import math
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from logger import logging_setup

# Setup for module
logger = logging_setup(__name__)
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")


def write_neg_edges(data: torch.tensor):
    """
    Write negative edges to file using filename from config.ini
    """
    # Write negative edges to file
    filename = config["FILENAMES"]["negative_samples"]
    filepath = Path(config["PATHS"]["neg_samples_path"])

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for edge in data:
            f.write(f"{edge[0]} {edge[1]}")
            f.write("\n")
    logger.info(f"Generated {filename}")
    return


def read_neg_edges() -> torch.tensor:
    """
    Read negative edges from file using filename from config.ini
    """
    filepath = config["PATHS"]["neg_samples_path"]
    # Read negative edges from file
    with open(filepath, "r") as f:
        lines = f.readlines()
        lines = [s.strip("\n") for s in lines]
        lines = [s.split(" ") for s in lines]
        lines = [[int(s[0]), int(s[1])] for s in lines]
        lines = torch.tensor(lines)
    filename = config["FILENAMES"]["negative_samples"]
    logger.info(f"Read {filename}")
    return lines


def active_abbrs() -> str:
    # Put parts together (in order) when activated
    parts = ""
    for part in config["DATASTRUCTURE"]:
        if "_abbreviation" in part:
            # remove _abbreviation from part
            part_bool = part.replace("_abbreviation", "")
            if config["DATASTRUCTURE"].getboolean(part_bool):
                parts += config["DATASTRUCTURE"][part]
    return parts


def get_n_from_config() -> int:
    if config["NUMBERINSTANCES"].getboolean("all"):
        n = "FULL DATA"
    else:
        n = config["NUMBERINSTANCES"].getint("number")
    return n


def positive_in_top_k_prob(x: int = 0, k: int = 50) -> float:
    """
    Calculate the expected value if edges are random
    """
    if config["STANDARD"]["graph_name"] == "ogbl-collab":
        x = 100000
        k = 50
    elif config["STANDARD"]["graph_name"] == "ogbl-ppa":
        x = 3000000
        k = 100
    else:
        return 0.0

    if x <= k - 1:
        return 1.0
    P = (math.comb(1, 1) * math.comb((x + 1) - 1, k - 1)) / math.comb(x + 1, k)
    return P


def delete_precomp_files() -> None:
    """
    Deletes the existing negative sample file for the active graph.
    """
    parts = active_abbrs()
    d_name = config["STANDARD"]["graph_name"]
    data_folder = config["PATHS"]["data_path"]
    files_deleted = 0

    for typ in ["train", "test", "valid"]:
        sparse_matrix_file = Path(
            data_folder, f"sparse_matrix_{typ}_{d_name}_{parts}.npz"
        )
        weights_file = Path(data_folder, f"weights_{typ}_{d_name}_{parts}.npy")

        # Delete the files if they exist
        if sparse_matrix_file.exists():
            sparse_matrix_file.unlink()
            files_deleted += 1
        if weights_file.exists():
            weights_file.unlink()
            files_deleted += 1

    groups_file = Path(data_folder, f"groups_{d_name}_{parts}.npy")

    # Delete the file if it exists
    if groups_file.exists():
        groups_file.unlink()
        files_deleted += 1

    # Delete the negative samples file if it exists
    negative_samples_file = Path(config["PATHS"]["neg_samples_path"])
    if negative_samples_file.exists():
        negative_samples_file.unlink()
        logger.info(f"Deleted {negative_samples_file}")

    # Log the number of files deleted
    logger.info(f"Deleted {files_deleted} precomputed files from type {d_name}_{parts}")

    return
