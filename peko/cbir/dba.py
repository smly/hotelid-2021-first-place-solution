import numpy as np

from .functions import l2norm_numpy


def qe_dba(
    feats_test, feats_index, sims, topk_idx, alpha=3.0, qe=True, dba=True
):
    # alpha-query expansion (DBA)
    feats_concat = np.concatenate([feats_test, feats_index], axis=0)

    weights = sims ** alpha
    feats_tmp = np.zeros(feats_concat.shape)
    for i in range(feats_concat.shape[0]):
        feats_tmp[i, :] = weights[i].dot(feats_concat[topk_idx[i]])

    del feats_concat
    feats_concat = l2norm_numpy(feats_tmp).astype(np.float32)

    split_at = [len(feats_test)]
    if qe and dba:
        return np.split(feats_concat, split_at, axis=0)
    elif not qe and dba:
        _, feats_index = np.split(feats_concat, split_at, axis=0)
        return feats_test, feats_index
    elif qe and not dba:
        feats_test, _ = np.split(feats_concat, split_at, axis=0)
        return feats_test, feats_index
    else:
        raise ValueError


def qe_dba_label_constrained(
    feats_test, feats_index, label_index,
    sims, topk_idx, alpha=3.0, qe=True, dba=True
):
    labels_concat = np.concatenate([
        # unlabeled data
        np.ones(feats_test.shape[0]) * -1,
        # labeled data
        label_index
    ], axis=0)
    feats_concat = np.concatenate([feats_test, feats_index], axis=0)
    assert labels_concat.shape[0] == feats_concat.shape[0]

    weights = sims ** alpha
    feats_tmp = np.zeros(feats_concat.shape)
    for i in range(feats_concat.shape[0]):
        if feats_test.shape[0] > i:
            # test images
            feats_tmp[i, :] = weights[i].dot(feats_concat[topk_idx[i]])
        else:
            # train images
            query_match = (labels_concat[topk_idx[i]] < 0) * 1.0
            binary_label_match = (labels_concat[topk_idx[i]] == labels_concat[i]) * 1.0
            weights_mask = (query_match + binary_label_match > 0.0) * 1.0
            label_constrained_weights = weights[i] * weights_mask
            feats_tmp[i, :] = label_constrained_weights.dot(feats_concat[topk_idx[i]])

    del feats_concat
    feats_concat = l2norm_numpy(feats_tmp).astype(np.float32)

    split_at = [len(feats_test)]
    if qe and dba:
        return np.split(feats_concat, split_at, axis=0)
    elif not qe and dba:
        _, feats_index = np.split(feats_concat, split_at, axis=0)
        return feats_test, feats_index
    elif qe and not dba:
        feats_test, _ = np.split(feats_concat, split_at, axis=0)
        return feats_test, feats_index
    else:
        raise ValueError
