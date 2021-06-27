import os

import cv2

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import gc
from logging import getLogger

import numpy as np
from peko.cbir.dba import qe_dba_label_constrained
from peko.cbir.search import predict_simple_mat, search_with_faiss_cpu
from peko.competition import (infer_test_rotations_with_v80model,
                              load_rf_models, make_normal_sub)
from peko.utils.logger import set_logger
from peko.utils.timer import timer

logger = getLogger("peko")


def v36(rot_csv):
    with timer("Load models", logger=logger):
        model_names = {
            "v114_regnety_augv5_s512_hotelidv2": 1.0,
            "v110_rs101e_augv5_s512_hotelidv2": 1.0,
            "v133_swin_augv5_s384_adamw_hotelidv3": 1.0,
            "v131_regnety_augv5_s512_hotelidv3": 1.0,
        }
        testdataset, feat_test, feat_index, label_index = load_rf_models(
            model_names, rot_csv, method="v3")

    with timer("DBAQE", logger=logger):
        if len(testdataset) != 3:
            n_qe, alpha = 5, 3.0
            feat_all = np.concatenate([feat_test, feat_index], axis=0)
            dists, topk_idx = search_with_faiss_cpu(feat_all, feat_all, topk=n_qe)
            feat_test, feat_index = qe_dba_label_constrained(
                feat_test, feat_index, label_index, dists, topk_idx, alpha=alpha, qe=True)

            feat_all = np.concatenate([feat_test, feat_index], axis=0)
            dists, topk_idx = search_with_faiss_cpu(feat_all, feat_all, topk=n_qe)
            feat_test, feat_index = qe_dba_label_constrained(
                feat_test, feat_index, label_index, dists, topk_idx, alpha=alpha, qe=True)

    with timer("Predict images ...", logger=logger):
        preds_mat = predict_simple_mat(feat_test, feat_index, label_index, n=100, topk=5)

    return testdataset, label_index, preds_mat


def v37(rot_csv):
    with timer("Load models", logger=logger):
        model_names = {
            "v114_regnety_augv5_s512_hotelidv2": 1.0,
            "v110_rs101e_augv5_s512_hotelidv2": 1.0,
            "v133_swin_augv5_s384_adamw_hotelidv3": 1.0,
            "v131_regnety_augv5_s512_hotelidv3": 1.0,
        }
        testdataset, feat_test, feat_index, label_index = load_rf_models(
            model_names, rot_csv, method="v4")

    with timer("Predict images ...", logger=logger):
        preds_mat = predict_simple_mat(feat_test, feat_index, label_index, n=100, topk=5)

    return testdataset, label_index, preds_mat


def numpy_topk(matrix, k, axis=0):
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort.take(np.arange(k), axis=axis)


def main():
    with timer("Infer image rotations", logger=logger):
        rot_csv = "/tmp/test_rotation_pred.csv"
        infer_test_rotations_with_v80model(rot_csv)
        gc.collect()

    with timer("v36", logger=logger):
        testdataset, label_index, preds_mat_v36 = v36(rot_csv)
        gc.collect()

    with timer("v37", logger=logger):
        _, _, preds_mat_v37 = v37(rot_csv)
        gc.collect()

    with timer("label mapping", logger=logger):
        label_idx_to_id = {idx: lbl for idx, lbl in enumerate(sorted(np.unique(label_index)))}
        preds_mat = preds_mat_v36 + preds_mat_v37
        preds_topk = numpy_topk(-preds_mat, 5, axis=1)
        preds = []
        for i in range(preds_mat.shape[0]):
            preds_top5 = []
            for j in range(5):
                lbl_idx = preds_topk[i, j]
                lbl = label_idx_to_id[lbl_idx]
                preds_top5.append(lbl)
            preds.append(preds_top5)

    del preds_mat
    make_normal_sub(preds, testdataset)


if __name__ == "__main__":
    set_logger(logger)
    main()
