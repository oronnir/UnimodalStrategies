# =============================================================================
# ECML-PKDD 2025 ‒ "Unimodal Strategies in Density-Based Clustering"
# Authors : Oron Nir, Jay Tenenbaum, Ariel Shamir
# Paper   : https://arxiv.org/abs/######   (pre-print link)
# Code    : https://github.com/oronnir/UnimodalStrategies
# License : MIT (see LICENSE file for full text)
# =============================================================================
import os.path
import pickle
import numpy as np
import traceback
from clustering import ternary_search_clustering
from evaluation import evaluate_solution
from logger_config import logger

seed = 42
np.random.seed(seed)


def load_ndarray_pkl(pkl_path):
    with open(pkl_path, 'rb') as file:
        x = pickle.load(file)
    assert isinstance(x, np.ndarray)
    logger.debug(f"Loaded data with shape {x.shape}.")
    return x


if __name__ == '__main__':
    # test clustering
    embeddings_pkl = r"C:\VI\FaceGroup\OtherDatasets\ESC-50\ESC-50-training-CLAP_vectors.pkl"

    # hyper params (user defined)
    min_pts = 5

    # load instances ND Array of shape [N, D]
    x = load_ndarray_pkl(embeddings_pkl)

    # run clustering
    pred_labels = []
    try:
        pred_labels, actual_k, best_solution = ternary_search_clustering(x, min_pts, k=None)
        label_set = set(pred_labels) - {-1}
        logger.info("Grouped into initial groups", extra={'num_clusters': len(label_set)})
    except Exception as e:
        logger.error(f"Clustering failure: {e}")
        traceback.print_exc()
        raise e

    # evaluate
    gt_labels_pkl = r"C:\VI\FaceGroup\OtherDatasets\ESC-50\ESC-50-training-CLAP_labels.pkl"

    # load class-labels, an ND Array of shape [N]
    gt_labels = load_ndarray_pkl(gt_labels_pkl)
    stats = evaluate_solution(pred_labels, gt_labels)
    logger.info(f"The evaluation concluded with the following statistics: {stats}")
