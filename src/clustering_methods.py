# =============================================================================
# ECML-PKDD 2025 ‒ "Unimodal Strategies in Density-Based Clustering"
# Authors : Oron Nir, Jay Tenenbaum, Ariel Shamir
# Paper   : https://arxiv.org/abs/######   (pre-print link)
# Code    : https://github.com/oronnir/UnimodalStrategies
# License : MIT (see LICENSE file for full text)
# =============================================================================
"""
The main method is the TSClustering from the paper and below a simplified version:

def ternary_search(x, lb, ub, minpts, iterCap) -> ClusteringSolution:
    '''
    ternary_search: find the best epsilon for DBSCAN clustering using modified ternary search.
    This is the simplified version of the ternary search from the paper.
    '''
    best = ClusteringSolution(None, lb, 0, 0, 0, None)
    for it in range(0, iterCap, 2):
        eps_range = (ub - lb) / 3
        ml = lb + eps_range
        mr = ub - eps_range

        # score ml/mr and update if better
        best, kl = cluster_and_update_best(best, x, minpts, ml, iterCap, it)
        best, kr = cluster_and_update_best(best, x, minpts, mr, iterCap, it+1)

        # update the search range based on the k
        if kl == 1 and kr == 1:
            # opt is on the left
            ub = ml
        elif kl > 1 and kr == 1:
            # opt is either left or middle
            ub = mr
        elif kl == 0 and kr == 1:
            # opt is in the middle
            lb = ml
            ub = mr
        elif kl <= 1 < kr:
            # opt is either right or middle
            lb = ml
        elif kl == 0 and kr == 0:
            # opt is on the right
            lb = mr
        elif kl > 1 and kr > 1:
            # case #6
            if kl > kr:
                ub = mr
            else:
                lb = ml

    # check the mid-point
    eps = (lb + ub) / 2
    best, _ = cluster_and_update_best(best, x, minpts, eps, iterCap, iterCap*2)
    return best
"""
import numpy as np
from sklearn.cluster import DBSCAN
from logger_config import logger


class ClusteringSolution:
    def __init__(self, cluster_ids: np.ndarray, epsilon: float, coverage: float, k: int, tsk_eps_ub: float=None, bin_epsilons=None, bin_ks=None):
        self.cluster_ids = cluster_ids
        self.epsilon = epsilon
        self.coverage = coverage
        self.k = k
        self.tsk_eps_ub = tsk_eps_ub
        self.bin_epsilons = bin_epsilons
        self.bin_ks = bin_ks

    def __repr__(self):
        base_stats = f'cluster_ids={None if self.cluster_ids is None else self.cluster_ids.shape}, epsilon={self.epsilon}, coverage={self.coverage}, k={self.k}'
        return f'ClusteringSolution({base_stats})'


def get_k_coverage(clusters: np.ndarray) -> (int, float):
    valid_clusters = clusters[clusters >= 0]
    cluster_sizes = np.bincount(valid_clusters)
    cluster_sizes = cluster_sizes[cluster_sizes > 1]
    actual_k = len(cluster_sizes)
    coverage = valid_clusters.shape[0] / clusters.shape[0]
    return actual_k, coverage


def solution_stats(clusters: np.ndarray):
    actual_k, coverage = get_k_coverage(clusters)
    return coverage, actual_k


def cluster_and_update_best(best_solution, x, min_samples, eps, max_iter, cur_iter) -> (ClusteringSolution, int):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, p=2.0, n_jobs=-1, metric='cosine').fit(x)
    clusters = clustering.labels_
    actual_k, coverage = get_k_coverage(clusters)

    # favoring large k
    if actual_k > best_solution.k:
        best_solution = ClusteringSolution(clusters, eps, coverage, actual_k)
        logger.debug(f'Improved best solution on iteration #{cur_iter + 1}/{max_iter + 1}: eps: {eps}, k: {actual_k}, coverage: {coverage}')
    return best_solution, actual_k


def ternary_search_dbscan(x, eps_lb: float, eps_ub: float, min_samples: int, num_iterations: int, k_prior: int = -1) -> ClusteringSolution:
    """
    ternary_search_dbscan: find the best epsilon for DBSCAN clustering using modified ternary search.
    We draw inspiration from the following algorithm: https://en.wikipedia.org/wiki/Ternary_search.
    :param x: feature vectors
    :param eps_ub: the upper bound of epsilon search range
    :param eps_lb: the initial epsilon to start the search from
    :param min_samples: the minimum number of samples in a cluster
    :param num_iterations: the maximum number of iterations to run the search
    :param k_prior: the prior knowledge of the number of clusters
    :return: the best solution
    """
    bin_ks = []
    bin_epsilons = []
    actual_k_right = 1
    best_solution = ClusteringSolution(None, eps_lb, 0, 0, 0, None)
    logger.debug('Starting ternary_search_dbscan', extra={'eps_lb': eps_lb, 'eps_ub': eps_ub, 'N': x.shape[0]})
    for iteration in range(0, num_iterations, 2):
        eps_range = (eps_ub - eps_lb) / 3
        mid_left = eps_lb + eps_range
        mid_right = eps_ub - eps_range
        logger.debug('Starting a loop over cluster_and_score calls', extra={'lower_bound': eps_lb, 'upper_bound': eps_ub, 'mid_left': mid_left, 'mid_right': mid_right})

        # score the two mid-points and update the best solution if better
        best_solution, actual_k_left = cluster_and_update_best(best_solution, x, min_samples, mid_left, num_iterations, iteration)
        best_solution, actual_k_right = cluster_and_update_best(best_solution, x, min_samples, mid_right, num_iterations, iteration + 1)

        bin_ks.append(actual_k_left)
        bin_ks.append(actual_k_right)
        bin_epsilons.append(mid_left)
        bin_epsilons.append(mid_right)

        # update the search range based on the k
        if actual_k_left == 1 and actual_k_right == 1:
            # the optimal solution is on the left side
            eps_ub = mid_left
        elif actual_k_left > 1 and actual_k_right == 1:
            # the optimal solution is either on the left side or in the middle
            eps_ub = mid_right
        elif actual_k_left == 0 and actual_k_right == 1:
            # the optimal solution is in the middle
            eps_lb = mid_left
            eps_ub = mid_right
        elif actual_k_left == 0 and actual_k_right > 1 or actual_k_left == 1 and actual_k_right > 1:
            # the optimal solution is either on the right side or in the middle
            eps_lb = mid_left
        elif actual_k_left == 0 and actual_k_right == 0:
            # the optimal solution is on the right side
            eps_lb = mid_right
        elif actual_k_left > 1 and actual_k_right > 1:
            # manage the case of a given k_prior
            if 0 < k_prior < actual_k_right:
                break

            # the optimal solution could be anywhere in the range so let's look at the scores
            if actual_k_left > actual_k_right:
                eps_ub = mid_right
            else:
                eps_lb = mid_left
        else:
            # two cases are assumed to be not applicable:
            # actual_k_left == 1 and actual_k_right == 0, or actual_k_left > 1 and actual_k_right == 0
            logger.warning('Unexpected case for ternary_search_dbscan - Ignoring the case and continuing!',
                         extra={'actual_k_left': actual_k_left, 'actual_k_right': actual_k_right})

    if k_prior <= 0 or k_prior < actual_k_right:
        # After max_steps, take the mid-point of the current range as the best solution
        epsilon = (eps_lb + eps_ub) / 2
        best_solution, actual_k_mid = cluster_and_update_best(best_solution, x, min_samples, epsilon, num_iterations, num_iterations * 2)

        bin_ks.append(actual_k_mid)
        bin_epsilons.append(epsilon)
    else:
        epsilon = actual_k_right

    # update the best solution with statistics
    coverage, actual_k = solution_stats(best_solution.cluster_ids)
    best_solution = ClusteringSolution(cluster_ids=best_solution.cluster_ids, epsilon=epsilon, coverage=coverage,
                                       k=actual_k, tsk_eps_ub=eps_ub, bin_epsilons=bin_epsilons, bin_ks=bin_ks)
    logger.debug('ternary_search_dbscan best results', extra={'best_k': best_solution.k,
          'best_eps': best_solution.epsilon, 'best_coverage': best_solution.coverage})
    return best_solution


def ternary_search_dbscan_with_k(x, eps_lb: float, eps_ub: float, min_samples: int, num_iterations: int, k_prior: int) -> ClusteringSolution:
    """
    ternary_search_dbscan: find the best epsilon for DBSCAN clustering using modified ternary search.
    We draw inspiration from the following algorithm: https://en.wikipedia.org/wiki/Ternary_search.
    ternary_search_dbscan_with_k runs ternary search with a given k_prior and stops when the k_prior is found unless the
     k_prior is not found. In that case, it runs a binary search to find the epsilon that yields the closest k.
    :param x: feature vectors
    :param eps_ub: the upper bound of epsilon search range
    :param eps_lb: the initial epsilon to start the search from
    :param min_samples: the minimum number of samples in a cluster
    :param num_iterations: the maximum number of iterations to run the search
    :param k_prior: the prior knowledge of the number of clusters
    :return: the best solution
    """
    ts_solution = ternary_search_dbscan(x, eps_lb, eps_ub, min_samples, num_iterations, k_prior)
    if ts_solution.k <= k_prior:
        return ts_solution

    # run binary search to find the epsilon that yields the closest k
    eps_ub = 1.0
    best_solution = ts_solution
    eps_lb = ts_solution.epsilon
    k_at_eps_ub = 1
    k_at_eps_lb = ts_solution.k
    EPS_TOLERANCE = 0.0000001

    # run binary search to find the epsilon that yields the closest k
    logger.debug('Starting binary search dbscan for k', extra={'eps_lb': eps_lb, 'eps_ub': eps_ub, 'N': x.shape[0]})
    iteration_counter = 1
    while k_prior != best_solution.k and k_at_eps_lb > k_at_eps_ub and eps_ub - eps_lb > EPS_TOLERANCE:
        logger.debug(f'iteration #{iteration_counter}: eps_lb: {eps_lb}, eps_ub: {eps_ub}, k_at_eps_lb: {k_at_eps_lb}, k_at_eps_ub: {k_at_eps_ub}')
        iteration_counter += 1
        mid_point = (eps_lb + eps_ub) / 2
        cluster_mid = DBSCAN(eps=mid_point, min_samples=min_samples, p=2.0, n_jobs=-1, metric='cosine').fit(x).labels_
        k_at_mid, coverage_mid = get_k_coverage(cluster_mid)
        if k_at_mid > k_prior:
            eps_lb = mid_point
            k_at_eps_lb = k_at_mid
        else:
            eps_ub = mid_point
            k_at_eps_ub = k_at_mid
        best_solution = ClusteringSolution(cluster_mid, mid_point, coverage_mid, k_at_mid)

    # update the best solution with statistics
    coverage, actual_k = solution_stats(best_solution.cluster_ids)
    best_solution = ClusteringSolution(best_solution.cluster_ids, best_solution.epsilon, coverage, actual_k)
    return best_solution


def ternary_search_estimator(x, eps_lb: float, eps_ub: float, min_samples: int, num_iterations: int) -> ClusteringSolution:
    """
    TSE: Run TS on alpha samples in both N and D to estimate the optimal epsilon for DBSCAN clustering without running
    on the full dataset.
    """
    # sample the data multiple times to estimate the optimal epsilon
    NUM_SAMPLES = 30
    ALPHA_D = 0.2
    ALPHA_N = 0.2
    sample_dim = [int(x.shape[0] * ALPHA_N), int(x.shape[1] * ALPHA_D)]
    eps_stars = []
    for i in range(NUM_SAMPLES):
        row_indices = np.random.choice(x.shape[0], sample_dim[0], replace=False)
        col_indices = np.random.choice(x.shape[1], sample_dim[1], replace=False)
        submatrix = x[row_indices][:, col_indices]

        # run ternary search on the submatrix
        best_solution = ternary_search_dbscan(submatrix, eps_lb, eps_ub, min_samples, num_iterations*4)
        eps_stars.append(best_solution.epsilon)

    # estimate the optimal epsilon
    LAMBDAS = np.array([0.4, 0.3, 0.3])
    eps_star = np.dot([eps_lb, eps_ub, np.mean(eps_stars)], LAMBDAS)

    # run DBSCAN on the full dataset
    best_solution = ClusteringSolution(None, eps_star, 0, 0, 0, None)
    best_solution = cluster_and_update_best(best_solution, x, min_samples, eps_star, num_iterations, num_iterations)
    return best_solution


def dip_unimodality_test(ex_ks):
    """
    dip_unimodality_test: run the dip test to determine the unimodality of the data
    :param ex_ks: the number of clusters
    :param ex_epsilons: the epsilons
    """
    import diptest
    dip = diptest.diptest(np.array(ex_ks))
    logger.info(f'Dip test result: {dip} over {len(ex_ks)} samples: DIP_stat={dip[0]} p_value={dip[1]}. If p_value >= 0.05, the data is unimodal!')
