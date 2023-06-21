"""
This module contains functions that generate matrices with data.
For each data part there is a specific function that can be called from this module.
"""

import torch
import math
from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp
from scipy.sparse import load_npz, save_npz, hstack, vstack, lil_matrix, csr_matrix
from configparser import ConfigParser, ExtendedInterpolation
from torch.utils.data import DataLoader

from logger import logging_setup

logger = logging_setup(__name__)

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")


def create_Emb_matrix(
    emb: torch.tensor, pos_edge: torch.tensor, neg_edge: torch.tensor
) -> ssp.csr_matrix:
    """
    Create Matrix with Node Embeddings for each Edge
    Returns Matrix with Dimension: 2*Embeddingsize X n_edges
    """
    print(type(emb), type(pos_edge), type(neg_edge))
    embedding_dim = len(emb[0])
    total_length = len(pos_edge) + len(neg_edge)
    edge_embeddings = np.empty((total_length, embedding_dim * 2))

    with tqdm(total=total_length, desc="Processing edges - Embeddings") as pbar:
        for i, edge in enumerate(pos_edge):
            edge_embeddings[i, :embedding_dim] = emb[edge[0] - 1].numpy()
            edge_embeddings[i, embedding_dim:] = emb[edge[1] - 1].numpy()
            pbar.update()

        for i, edge in enumerate(neg_edge):
            edge_embeddings[i + len(pos_edge), :embedding_dim] = emb[
                edge[0] - 1
            ].numpy()
            edge_embeddings[i + len(pos_edge), embedding_dim:] = emb[
                edge[1] - 1
            ].numpy()
            pbar.update()

    edge_data_sparse = ssp.csr_matrix(edge_embeddings)

    return edge_data_sparse


def create_Nid_matrix(
    pos_edge: torch.tensor, neg_edge: torch.tensor, n_nodes: int
) -> ssp.csr_matrix:
    """
    Create Matrix with ones for edge nodes
    Returns Matrix with Dimension: n_nodes X n_edges
    """
    total_edges = len(pos_edge) + len(neg_edge)

    # Create an empty sparse matrix
    matrix = lil_matrix((total_edges, 2 * n_nodes), dtype=int)

    with tqdm(total=total_edges, desc="Processing edges - Node IDs") as pbar:
        # Process pos_edge
        for i, edge in enumerate(pos_edge):
            matrix[i, edge[0] - 1] = 1
            matrix[i, n_nodes + edge[1] - 1] = 1
            pbar.update()

        # Process neg_edge
        for i, edge in enumerate(neg_edge):
            matrix[i + len(pos_edge), edge[0] - 1] = 1
            matrix[i + len(pos_edge), n_nodes + edge[1] - 1] = 1
            pbar.update()

    return matrix


def create_Ngh_matrix(
    pos_edge: torch.tensor,
    neg_edge: torch.tensor,
    A: ssp.csr_matrix,
    binary: bool = False,
) -> ssp.csr_matrix:
    """
    Create Matrix with Neighboorhod vectors
    if binary: Neighbors = 1 else Neighbors = n_connections
    Returns Matrix with Dimension: 2*n_nodes X n_edges
    """
    matrix_rows = []
    edges = torch.cat((pos_edge, neg_edge))
    for edge in tqdm(edges, desc="Neighborhood Data Creation"):
        u, v = edge[0], edge[1]
        row = hstack((A[u], A[v]))
        if binary:
            row.data[row.data != 0] = 1
        matrix_rows.append(row)
    matrix = vstack(matrix_rows)
    print(matrix.shape)
    return matrix


def create_SumAdamic_matrix(
    pos_edge: torch.tensor,
    neg_edge: torch.tensor,
    A: ssp.csr_matrix,
    batch_size: int = 100000,
) -> ssp.csr_matrix:
    """
    Create Matrix with Adamic Adar Sum for each Edge
    Returns Matrix with Dimension: 1 X n_edges
    """
    edges = torch.cat((pos_edge, neg_edge)).t()
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edges.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edges[0, ind], edges[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    sparse_matrix = csr_matrix(scores.reshape(-1, 1))
    return sparse_matrix


def create_RA_matrix(
    pos_edge: torch.tensor,
    neg_edge: torch.tensor,
    A: ssp.csr_matrix,
    batch_size: int = 32768,
) -> ssp.csr_matrix:
    """
    Create Matrix with Resource Allocation index for each Edge
    Returns Matrix with Dimension: 1 X n_edges
    """

    # cite: [Predicting missing links via local information](https://arxiv.org/pdf/0901.0553.pdf)
    # Code adapted from: https://github.com/CUAI/Edge-Proposal-Sets

    edges = torch.cat((pos_edge, neg_edge))
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    D = A.multiply(w).tocsr()  # e[i,j] / log(d_j)

    link_index = edges.t()
    link_loader = DataLoader(range(link_index.size(1)), batch_size)
    scores = []
    for idx in tqdm(link_loader):
        src, dst = link_index[0, idx], link_index[1, idx]
        batch_scores = np.array(np.sum(A[src].multiply(D[dst]), 1)).flatten()
        scores.append(batch_scores)
    scores = np.concatenate(scores, 0)
    sparse_matrix = csr_matrix(scores.reshape(-1, 1))
    return sparse_matrix


def create_CN_array(
    pos_edge: torch.tensor,
    neg_edge: torch.tensor,
    A: ssp.csr_matrix,
    batch_size: int = 100000,
) -> ssp.csr_matrix:
    """
    Create Matrix with Number of common neighbors for each Edge
    Returns Matrix with Dimension: 1 X n_edges
    """
    # Absolute amount of neighbors
    edges = torch.cat((pos_edge, neg_edge)).t()
    link_loader = DataLoader(range(edges.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edges[0, ind], edges[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    sparse_matrix = csr_matrix(scores.reshape(-1, 1))
    return sparse_matrix


def create_CN_matrix(
    pos_edge: torch.tensor, neg_edge: torch.tensor, A: ssp.csr_matrix, binary=False
) -> ssp.csr_matrix:
    """
    Create Matrix with Common Neighbors for each Edge
    if Binary: Common neighbors = 1 else CN = A[u,x]*A[v,x]
    Returns Matrix with Dimension: n_nodes X n_edges
    """
    matrix_rows = []
    edges = torch.cat((pos_edge, neg_edge))
    for edge in tqdm(edges, desc="Common Neighborhood Data Creation"):
        u, v = edge[0], edge[1]

        # Get common neighbors for nodes u and v
        row = A[u].multiply(A[v])

        if binary:
            row[row != 0] = 1

        matrix_rows.append(row)
    matrix = vstack(matrix_rows)
    return matrix


def create_CNA_matrix(
    pos_edge: torch.tensor, neg_edge: torch.tensor, A: ssp.csr_matrix
) -> ssp.csr_matrix:
    """
    Create Matrix with normalized Common Neighbors for each Edge
    Each CN is normalized by sqrt(sum_CN)
    Returns Matrix with Dimension: n_nodes X n_edges
    """

    matrix_rows = []
    edges = torch.cat((pos_edge, neg_edge))
    for edge in tqdm(edges, desc="Common Neighborhood Data Creation"):
        u, v = edge[0], edge[1]

        # Get common neighbors for nodes u and v
        common_neighbors = A[u].multiply(A[v])

        # Calculate the sum of existing common neighbors
        common_sum = common_neighbors.sum()

        # If no common neighbors, add an empty row
        if common_sum == 0:
            row = csr_matrix(
                (1, A.shape[1])
            )  # create an empty row with the same number of columns as A
        else:
            # Normalize each common neighbor by the sum of existing common neighbors
            common_normalized = common_neighbors / math.sqrt(common_sum)

            # Append the normalized common neighbors to the list of matrix rows
            row = common_normalized

        matrix_rows.append(row)
    matrix = vstack(matrix_rows)
    print(matrix.shape)
    return matrix


def create_CNB_matrix(
    pos_edge: torch.tensor, neg_edge: torch.tensor, A: ssp.csr_matrix
) -> ssp.csr_matrix:
    """
    Create Matrix with normalized Common Neighbors for each Edge
    Each CN is normalized by sqrt(degree of CN)
    Returns Matrix with Dimension: n_nodes X n_edges
    """

    # Create the graph from the adjacency matrix
    total_edges = len(pos_edge) + len(neg_edge)

    # Create sparse matrix of the same size as A
    # Create an empty sparse matrix
    CNB = lil_matrix((total_edges, A.shape[0]), dtype=np.float32)
    edges = torch.cat((pos_edge, neg_edge))
    # Fill CNB matrix
    for i, edge in tqdm(
        enumerate(edges), desc="Common Neighborhood Data Creation", total=len(edges)
    ):
        u, v = edge[0], edge[1]
        common_neigh = A[u].multiply(A[v]).nonzero()[1]

        for neigh in common_neigh:
            degree = A[neigh].sum()
            if degree != 0:
                CNB[i, neigh] = 1 / np.sqrt(degree)

    return CNB


def create_NnA_matrix(
    pos_edge: torch.tensor, neg_edge: torch.tensor, A: ssp.csr_matrix
) -> ssp.csr_matrix:
    """
    Create Matrix with normalized Neighbors for each Edge
    Each neighbor is normalized by sqrt(sum_Neighbors)
    Returns Matrix with Dimension: 2*n_nodes X n_edges
    """
    matrix_rows = []
    edges = torch.cat((pos_edge, neg_edge))
    for edge in tqdm(edges, desc="Neighborhood Data Creation"):
        u, v = edge[0], edge[1]  # Subtract 1 for zero-based indexing

        # Calculate the sum of existing neighbors for nodes u and v
        u_sum = A[u].sum()
        v_sum = A[v].sum()

        # Normalize each neighbor by the sum of existing neighbors
        u_normalized = A[u] / math.sqrt(u_sum)
        v_normalized = A[v] / math.sqrt(v_sum)

        # Concatenate the normalized neighbors
        row = hstack((u_normalized, v_normalized))
        matrix_rows.append(row)
    matrix = vstack(matrix_rows)
    print(matrix.shape)
    return matrix


def create_NnB_matrix(
    pos_edge: torch.tensor, neg_edge: torch.tensor, A: ssp.csr_matrix
) -> ssp.csr_matrix:
    """
    Create Matrix with normalized Neighbors for each Edge
    Each neighbors is normalized by sqrt(sum_CN)
    Returns Matrix with Dimension: 2*n_nodes X n_edges
    """
    matrix_rows = []
    edges = torch.cat((pos_edge, neg_edge))
    for edge in tqdm(edges, desc="Neighborhood Data Creation"):
        u, v = edge[0] - 1, edge[1] - 1  # Subtract 1 for zero-based indexing

        # Find the common neighbors of nodes u and v
        common_neighbors = set(A[u].nonzero()[1]).intersection(set(A[v].nonzero()[1]))

        # Calculate the sum of common neighbors
        common_neighbor_sum = (
            A[u, common_neighbors].sum() + A[v, common_neighbors].sum()
        )

        # Normalize each common neighbor by the sum of common neighbors
        u_normalized = A[u] / math.sqrt(common_neighbor_sum)
        v_normalized = A[v] / math.sqrt(common_neighbor_sum)

        # Concatenate the normalized neighbors
        row = hstack((u_normalized, v_normalized))
        matrix_rows.append(row)
    matrix = vstack(matrix_rows)
    print(matrix.shape)
    return matrix


def create_Jaccard_array(
    pos_edge: torch.tensor, neg_edge: torch.tensor, A: ssp.csr_matrix
) -> ssp.csr_matrix:
    """
    Create Matrix with Jaccard Index for each Edge
    Returns Matrix with Dimension: 2 X n_edges
    """
    jaccard_scores = []
    edges = torch.cat((pos_edge, neg_edge))
    for edge in tqdm(edges, desc="Jaccard Index Data Creation"):
        u, v = edge[0], edge[1]

        # Get neighbors for nodes u and v
        neighbors_u = A[u].indices
        neighbors_v = A[v].indices

        # Compute the Jaccard index
        intersection = len(np.intersect1d(neighbors_u, neighbors_v))
        union = len(np.union1d(neighbors_u, neighbors_v))
        jaccard_index = intersection / union if union != 0 else 0

        jaccard_scores.append(jaccard_index)

    # Convert list to a 1D numpy array
    jaccard_scores = np.array(jaccard_scores).reshape(-1, 1)

    # Convert numpy array to sparse matrix
    jaccard_scores_sparse = csr_matrix(jaccard_scores)

    return jaccard_scores_sparse
