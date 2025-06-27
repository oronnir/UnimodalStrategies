# =============================================================================
# ECML-PKDD 2025 ‒ “Unimodal Strategies in Density-Based Clustering”
# Authors : Oron Nir, Jay Tenenbaum, Ariel Shamir
# Paper   : https://arxiv.org/abs/######   (pre-print link)
# Code    : https://github.com/oronnir/UnimodalStrategies
# License : MIT (see LICENSE file for full text)
# =============================================================================
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def evaluate_solution(pred, gt):
    """
    Return the NMI, ARI, and Noise
    Note: NMI and ARI are reported on the non-noisy prediction labels.
    """
    n = len(gt)
    nmi = normalized_mutual_info_score(gt, pred)
    valid_labels = [topic_id for topic_id, cluster_id in zip(list(gt), pred) if cluster_id != -1]
    valid_ordered_cluster_labels = [cluster_id for cluster_id in pred if cluster_id != -1]
    nmi_without_noise = normalized_mutual_info_score(valid_labels, valid_ordered_cluster_labels)
    ari = adjusted_rand_score(list(gt), pred)
    ari_without_noise = adjusted_rand_score(valid_labels, valid_ordered_cluster_labels)
    noise = 1 - len(valid_ordered_cluster_labels)/n
    return dict(n=n, noise=noise, nmi=nmi, nmi_without_noise=nmi_without_noise, ari=ari, ari_without_noise=ari_without_noise)
