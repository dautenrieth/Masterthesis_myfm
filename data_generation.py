"""
This module is used to read or create the data.
Additionally, two auxiliary functions are defined,
which are needed for the generation.
The individual data generation functions are called from the data_parts module.
"""

import time
import torch
import random
import os, math
from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp
from typing import Tuple
from scipy.sparse import load_npz, save_npz, hstack, vstack, lil_matrix, csr_matrix
import utils as ut
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation
from ogb.linkproppred import PygLinkPropPredDataset
import networkx as nx
from torch.utils.data import DataLoader
import data_parts as dp

from logger import logging_setup

logger = logging_setup(__name__)

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")


def get_data(typ: str = "") -> Tuple[ssp.csr_matrix, np.ndarray, list]:
    """
    Reads in or generates data.
    If files dont exist data will be generated based on config.
    Returns:- matrix with data
            - vector with weights
            - list of groups
    """
    parts = ut.active_abbrs()
    d_name = config["STANDARD"]["graph_name"]
    data_folder = config["PATHS"]["data_path"]

    sparse_matrix_file = Path(data_folder, f"sparse_matrix_{typ}_{d_name}_{parts}.npz")
    weights_file = Path(data_folder, f"weights_{typ}_{d_name}_{parts}.npy")
    groups_file = Path(data_folder, f"groups_{d_name}_{parts}.npy")

    # Check if data files exist (if activated with groups file)
    if (
        os.path.exists(sparse_matrix_file)
        and os.path.exists(weights_file)
        and config["DATASTRUCTURE"].getboolean("Grouping")
        and os.path.exists(groups_file)
    ):
        logger.debug("Start reading in files")
        # if file exist, then loead Sparse-Matrix and Array
        edge_data_sparse = load_npz(sparse_matrix_file)
        weights = np.load(weights_file)
        groups = np.load(groups_file)
        logger.info(f"Read in {sparse_matrix_file}, {weights_file} and {groups_file}")

    elif (
        os.path.exists(sparse_matrix_file)
        and os.path.exists(weights_file)
        and not config["DATASTRUCTURE"].getboolean("Grouping")
    ):
        logger.debug("Start reading in files")
        # if file exist, then load Sparse-Matrix and Array
        edge_data_sparse = load_npz(sparse_matrix_file)
        weights = np.load(weights_file)
        logger.info(f"Read in {sparse_matrix_file} and {weights_file}")
    else:
        # if files dont exist, generate data
        logger.debug("No fitting files found - start creation")
        logger.debug(f"Searched for {sparse_matrix_file}, {weights_file}")
        if config["DATASTRUCTURE"].getboolean("Grouping"):
            logger.debug("and grouping file")

        dataset = PygLinkPropPredDataset(name=d_name)
        data = dataset[0]
        emb = dataset[0]["x"]
        # split loaded data
        split_edges = dataset.get_edge_split()
        train_edges, valid_edges, test_edges = (
            split_edges["train"],
            split_edges["valid"],
            split_edges["test"],
        )
        pweights = None
        if typ == "train":
            if "weight" in train_edges:
                pweights = train_edges["weight"]
            pos_edge = train_edges["edge"]
            # Check if file with negative samples exists
            filepath = config["PATHS"]["neg_samples_path"]
            if os.path.exists(filepath):
                logger.debug("Start reading in negative edges")
                neg_samples = ut.read_neg_edges()
                if len(neg_samples) != len(train_edges["edge"]):
                    # Create negative samples
                    neg_samples = create_neg_samples(
                        train_edges,
                        valid_edges,
                        test_edges,
                        number_of_nodes=data.num_nodes,
                    )
                    # Write negative samples to file
                    ut.write_neg_edges(neg_samples)
            else:
                # Create negative samples
                neg_samples = create_neg_samples(
                    train_edges, valid_edges, test_edges, number_of_nodes=data.num_nodes
                )
                # Write negative samples to file
                ut.write_neg_edges(neg_samples)

            neg_edge = neg_samples
        elif typ == "valid":
            pos_edge = valid_edges["edge"]
            if "weight" in valid_edges:
                pweights = valid_edges["weight"]
            neg_edge = valid_edges["edge_neg"]
        elif typ == "test":
            pos_edge = test_edges["edge"]
            if "weight" in test_edges:
                pweights = test_edges["weight"]
            neg_edge = test_edges["edge_neg"]
        else:
            logger.error(
                f"Broke Naming Convention of sparse matrix: {typ}. Should be either train, valid or test"
            )

        if config["DATASTRUCTURE"].getboolean("Grouping"):
            groups = []
        else:
            groups = None

        # Start with empty matrix and then stack to it
        edge_data_sparse = ssp.csr_matrix((len(pos_edge) + len(neg_edge), 0))

        # Check every implemented datapart
        if config["DATASTRUCTURE"].getboolean("Embeddings"):
            eds_tmp = dp.create_Emb_matrix(emb, pos_edge, neg_edge)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("NodeIDs"):
            eds_tmp = dp.create_Nid_matrix(pos_edge, neg_edge, data.num_nodes)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        # Methods that need the adjacency matrix
        if (
            config["DATASTRUCTURE"].getboolean("Adamic_Adar_Sum")
            or config["DATASTRUCTURE"].getboolean("Jaccard")
            or config["DATASTRUCTURE"].getboolean("Neighborhood")
            or config["DATASTRUCTURE"].getboolean("Neighborhood_Binary")
            or config["DATASTRUCTURE"].getboolean("Resource_Allocation")
            or config["DATASTRUCTURE"].getboolean("Common_Neighborhood")
            or config["DATASTRUCTURE"].getboolean("Common_Neighborhood_Binary")
            or config["DATASTRUCTURE"].getboolean("Common_Neighborhood_Int")
            or config["DATASTRUCTURE"].getboolean("Common_Ngh_NormA")
            or config["DATASTRUCTURE"].getboolean("Common_Ngh_NormB")
            or config["DATASTRUCTURE"].getboolean("Neighborhood_NormA")
            or config["DATASTRUCTURE"].getboolean("Neighborhood_NormB")
        ):
            A = construct_graph(data, train_edges)

        if config["DATASTRUCTURE"].getboolean("Neighborhood"):
            eds_tmp = dp.create_Ngh_matrix(pos_edge, neg_edge, A, binary=False)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Neighborhood_Binary"):
            eds_tmp = dp.create_Ngh_matrix(pos_edge, neg_edge, A, binary=True)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Adamic_Adar_Sum"):
            eds_tmp = dp.create_SumAdamic_matrix(pos_edge, neg_edge, A)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Jaccard"):
            eds_tmp = dp.create_Jaccard_array(pos_edge, neg_edge, A)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Resource_Allocation"):
            eds_tmp = dp.create_RA_matrix(pos_edge, neg_edge, A)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Common_Neighborhood"):
            eds_tmp = dp.create_CN_array(pos_edge, neg_edge, A)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Common_Neighborhood_Binary"):
            eds_tmp = dp.create_CN_matrix(pos_edge, neg_edge, A, binary=True)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Common_Neighborhood_Int"):
            eds_tmp = dp.create_CN_matrix(pos_edge, neg_edge, A, binary=False)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Common_Ngh_NormA"):
            eds_tmp = dp.create_CNA_matrix(pos_edge, neg_edge, A)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Common_Ngh_NormB"):
            eds_tmp = dp.create_CNB_matrix(pos_edge, neg_edge, A)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Neighborhood_NormA"):
            eds_tmp = dp.create_NnA_matrix(pos_edge, neg_edge, A)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        if config["DATASTRUCTURE"].getboolean("Neighborhood_NormB"):
            eds_tmp = dp.create_NnB_matrix(pos_edge, neg_edge, A)
            if config["DATASTRUCTURE"].getboolean("Grouping"):
                groups.append(eds_tmp.shape[1])
            edge_data_sparse = hstack([edge_data_sparse, eds_tmp])

        sparse_matrix_file.parent.mkdir(parents=True, exist_ok=True)

        save_npz(sparse_matrix_file, edge_data_sparse)
        logger.info(f"Saved matrix as {sparse_matrix_file}")

        # Create weights
        if config["RUNS"].getboolean("set_edge_weights_one"):
            pos_weights = np.ones(len(pos_edge), dtype=int)
        else:
            if pweights is not None:
                pos_weights = pweights
            else:
                pos_weights = np.ones(len(pos_edge), dtype=int)
        neg_weights = np.zeros(len(neg_edge), dtype=int)

        # Kombiniere die beiden Arrays
        weights = np.concatenate((pos_weights, neg_weights))

        # Speichern Sie den Array als Datei
        np.save(weights_file, weights)
        logger.info(f"Saved weigths as {weights_file}")

        if config["DATASTRUCTURE"].getboolean("Grouping"):
            np.save(groups_file, groups)
            logger.info(f"Saved groups as {groups_file}")
    return edge_data_sparse, weights, groups


def construct_graph(data: any, train_edges: dict) -> ssp.csr_matrix:
    """
    Creating an Adjacency Matrix based training edges and edge weights
    """
    logger.debug("Constructing graph.")
    train_edges_raw = np.array(train_edges["edge"])
    train_edges_reverse = np.array(
        [train_edges_raw[:, 1], train_edges_raw[:, 0]]
    ).transpose()
    train_edges = np.concatenate([train_edges_raw, train_edges_reverse], axis=0)
    edge_weight = torch.ones(train_edges.shape[0], dtype=int)
    A = ssp.csr_matrix(
        (edge_weight, (train_edges[:, 0], train_edges[:, 1])),
        shape=(data.num_nodes, data.num_nodes),
    )
    return A


def create_neg_samples(
    train_edges: dict,
    valid_edges: dict,
    test_edges: dict,
    number_of_nodes: int = None,
    number_of_neg_samples: int = None,
) -> torch.tensor:
    """
    Creating negatives edge samples based on exisiting eges
    """

    start_time = time.time()

    # Merge all edges
    merged_tensor = torch.cat(
        (
            train_edges["edge"],
            valid_edges["edge"],
            test_edges["edge"],
            valid_edges["edge_neg"],
            test_edges["edge_neg"],
        ),
        dim=0,
    )
    # Convert tensor to set
    merged_tensor = set(map(tuple, merged_tensor.numpy()))

    # Get negative edges when not in train, valid or test
    if number_of_neg_samples is None:
        number_of_neg_samples = len(
            train_edges["edge"]
        )  # Equal to number of positive edges
    neg_samples = set()
    for index in tqdm(range(number_of_neg_samples), desc="Creating negative edges"):
        # edge = [node1, node2]
        # Create random edge
        # Generate two random integers between 1 and number_of_nodes
        node1 = random.randint(0, number_of_nodes - 1)
        node2 = random.randint(0, number_of_nodes - 1)

        # Check if edge is not in merged_tensor nor in neg_samples
        while (node1, node2) in merged_tensor or (node1, node2) in neg_samples:
            node1 = random.randint(0, number_of_nodes - 1)
            node2 = random.randint(0, number_of_nodes - 1)
        # Add edge to neg_samples
        neg_samples.add((node1, node2))

    # Convert set to tensor
    neg_samples = torch.tensor(list(neg_samples))
    # Log the elapsed time as a message
    logger.info(
        f"create_neg_samples-function execution time: {time.time() - start_time:.2f} seconds for {number_of_neg_samples} edges"
    )
    return neg_samples
