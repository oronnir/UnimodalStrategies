# =============================================================================
# ECML-PKDD 2025 ‒ "Unimodal Strategies in Density-Based Clustering"
# Authors : Oron Nir, Jay Tenenbaum, Ariel Shamir
# Paper   : https://arxiv.org/abs/######   (pre-print link)
# Code    : https://github.com/oronnir/UnimodalStrategies
# License : MIT (see LICENSE file for full text)
# =============================================================================

import time
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cosine
from clustering_methods import ternary_search_dbscan, get_k_coverage, ternary_search_dbscan_with_k
from logger_config import logger


def init_epsilon_ub(x, eps_lb, eps_ub, min_samples, alpha=0.2, num_iterations=10):
    """
    Initialize the epsilon search by running DBSCAN with the given eps_lb and eps_ub.
    :param x: the feature vector per box
    :param min_samples: the minimum number of boxes in a cluster
    :param eps_lb: the lower bound of the epsilon search
    :param eps_ub: the upper bound of the epsilon search
    :param alpha: the dimensionality reduction factor
    :param num_iterations: the number of iterations for the ternary search
    :return: the best solution found
    """
    # sample 20% of the data to speed up the search
    n_samples = int(x.shape[0] * alpha)
    sampled_row_indices = np.random.choice(x.shape[0], size=n_samples, replace=False)
    best_solution = ternary_search_dbscan(x[sampled_row_indices, :], eps_lb=eps_lb, eps_ub=eps_ub, min_samples=min_samples, num_iterations=num_iterations)
    if best_solution is None or best_solution.epsilon is None:
        logger.error('Clustering failed on initialization')
        return eps_ub
    return best_solution.epsilon


def init_epsilon_lb(x, eps_lb, eps_ub, min_cluster_size, alpha, num_iterations=10):
    """
    Initialize the epsilon lower bound by running DBSCAN TernarySearch on a lower dimension problem and bounded space.
    :param x: the feature vector per box
    :param eps_lb: the lower bound of the epsilon search
    :param eps_ub: the upper bound of the epsilon search
    :param min_cluster_size: the minimum number of boxes in a cluster
    :param alpha: the dimensionality reduction factor
    :param num_iterations: the number of iterations for the ternary search
    :return: the lower bound for epsilon on the larger problem
    """
    # take the top variance features
    variances = np.var(x, axis=0)
    top_variance_indices = np.argsort(variances)[::-1]

    # take the bottom variance features
    top_variance_indices = top_variance_indices[-int(x.shape[1] * alpha):]

    # project the data
    x = x[:, top_variance_indices]

    # run the ternary search
    best_solution = ternary_search_dbscan(x, eps_lb=eps_lb, eps_ub=eps_ub, min_samples=min_cluster_size, num_iterations=num_iterations)
    if best_solution is None or best_solution.epsilon is None:
        logger.error('Clustering failed on initialization')
        return eps_lb
    return best_solution.epsilon


def init_dbscan_hyper_parameters(x, min_cluster_size):
    # adjust the hyperparameters by the scale of the input
    # set the max_iterations w.r.t. the number of samples
    iterations = 12
    eps_ub = 1.0
    eps_lb = 0.0000001
    alpha = 0.2

    start_time = time.time()
    eps_ub = min(eps_ub, init_epsilon_ub(x, eps_lb, eps_ub, min_cluster_size, alpha=alpha, num_iterations=iterations))
    logger.debug('init_epsilon_ub', extra={'duration': time.time() - start_time})
    eps_lb = max(eps_lb, init_epsilon_lb(x, eps_lb, eps_ub, min_cluster_size, alpha=alpha, num_iterations=iterations))
    logger.debug('init_epsilon_lb + UB', extra={'duration': time.time() - start_time})
    return iterations, eps_lb, eps_ub


def ternary_search_clustering(features, min_cluster_size, k=None) -> (np.ndarray, int, np.ndarray, dict):
    """
    Ternary search for the right eps parameter by DBSCAN.
    Merging concepts from https://en.wikipedia.org/wiki/Ternary_search
    :param features: the feature vector per sample an array of shape (N, D)
    :param min_cluster_size: the minimum number of samples in a cluster. 2 is a good value if the representation is strong.
    :param k: the number of clusters to estimate. If None, the algorithm will estimate the number of clusters.
    :return: cluster_ids, k_estimate
    """
    actual_input_size = features.shape[0]
    chromatic_ids = np.arange(actual_input_size)
    default_degenerate_solution = chromatic_ids, actual_input_size, chromatic_ids, dict(zip(chromatic_ids, np.zeros([actual_input_size], dtype=int)))

    # init hyperparameters
    iterations, eps_lb, eps_ub = init_dbscan_hyper_parameters(features, min_cluster_size)

    # cluster the data
    if k is None:
        best_solution = ternary_search_dbscan(features, eps_lb=eps_lb, eps_ub=eps_ub, min_samples=min_cluster_size, num_iterations=iterations)
    else:
        best_solution = ternary_search_dbscan_with_k(features, eps_lb=eps_lb, eps_ub=eps_ub, min_samples=min_cluster_size, num_iterations=iterations, k_prior=k)

    if best_solution is None or best_solution.cluster_ids is None:
        logger.error('Clustering failed', extra={'actual_input_size': actual_input_size})
        return default_degenerate_solution

    logger.info('cast_clustering statistics', extra={'k_estimate': best_solution.k, 'p_noise': best_solution.coverage, 'actual_DBSCAN_n_clusters': best_solution.k})
    cluster_ids = best_solution.cluster_ids

    # get the best thumbnail per cluster
    actual_k, coverage = get_k_coverage(cluster_ids)
    logger.info('Clustering final statistics', extra={'actual_k_final': actual_k, 'coverage_final': coverage})
    return cluster_ids, actual_k, best_solution
