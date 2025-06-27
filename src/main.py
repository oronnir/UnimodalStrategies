# =============================================================================
# ECML-PKDD 2025 ‒ “Unimodal Strategies in Density-Based Clustering”
# Authors : Oron Nir*, Jay Tenenbaum, Ariel Shamir
# Paper   : https://arxiv.org/abs/######   (pre-print link)
# Code    : https://github.com/oronnir/UnimodalStrategies
# License : MIT (see LICENSE file for full text)
# =============================================================================
import os.path
import pickle
import numpy as np
import traceback
from clustering import ternary_search_clustering

seed = 42
np.random.seed(seed)


def run_clustering_test(pkl_path, k=None):
    # hyper params
    min_pts = 5

    # load data
    with open(pkl_path, 'rb') as file:
        x = pickle.load(file)

    try:
        assert isinstance(x, np.ndarray)
        print(f"Loaded data with shape {x.shape} of N={x.shape[0]} samples and D={x.shape[1]} features.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        traceback.print_exc()
        raise e

    # run clustering
    labels = []
    try:
        labels, actual_k, best_solution = ternary_search_clustering(x, min_pts, k)
        label_set = set(labels) - {-1}
        print("Grouped into initial groups", dict(num_clusters=len(label_set)))
    except Exception as e:
        print(f"Clustering failure: {e}")
        traceback.print_exc()

    return labels


if __name__ == '__main__':
    # test clustering
    dataset_path = r"C:\...\ESC.pkl"
    dataset_known_k = None

    # run clustering test
    run_clustering_test(dataset_path, dataset_known_k)
    print('Clustering test passed!')
