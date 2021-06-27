from collections import defaultdict
from logging import getLogger

import faiss
import ml_metrics
import numpy as np
import torch
from peko.cbir.functions import l2norm_numpy
from peko.utils.timer import timer

logger = getLogger("peko")


def predict_simple_mat(feat_test, feat_index, label_index, n=150, topk=50):
    label_lbl_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(np.unique(label_index)))}
    n_labels = len(np.unique(label_index))
    preds_mat = np.zeros((feat_test.shape[0], n_labels), dtype=np.float32)

    dists, topk_idx = search_with_faiss_cpu(feat_test, feat_index, topk=n)
    for i in range(dists.shape[0]):
        pred_class = []
        for k, j in enumerate(label_index[topk_idx[i]]):
            if j in pred_class:
                continue
            d = dists[i, k]
            preds_mat[i, label_lbl_to_idx[j]] = d
            pred_class.append(j)

    return preds_mat


def predict_simple(feat_test, feat_index, label_index, n=100, topk=5):
    dists, topk_idx = search_with_faiss_cpu(feat_test, feat_index, topk=n)
    ret = []
    for i in range(dists.shape[0]):
        pred_class = []
        for j in label_index[topk_idx[i]]:
            if j in pred_class:
                continue
            pred_class.append(j)
            if len(pred_class) == topk:
                break
        ret.append(pred_class)

    return ret


def predict_greedy_sum(feat_test, feat_index, label_index, n=100, topk=5):
    dists, topk_idx = search_with_faiss_cpu(feat_test, feat_index, topk=n)

    ret = []
    for i in range(dists.shape[0]):
        sims = defaultdict(int)
        pred_class = []
        for nbr_i, j in enumerate(label_index[topk_idx[i]]):
            sims[j] += dists[i, nbr_i]
            if j in pred_class:
                continue
            pred_class.append(j)
            if len(pred_class) == topk:
                break

        pred_class = []
        for lbl, _ in list(reversed(sorted(sims.items(), key=lambda x: x[1]))):
            pred_class.append(lbl)
        ret.append(pred_class)

    return ret


def compute_mapk_score(label_val, label_index, dists, topk_idx, topk=5, unique=False):
    apk_scores = []
    for i in range(dists.shape[0]):
        # Naive methods
        pred_class = label_index[topk_idx[i]]
        if not unique:
            pred_class = pred_class[:topk]
        else:
            pred_class = []
            for j in label_index[topk_idx[i]]:
                if j in pred_class:
                    continue
                pred_class.append(j)
                if len(pred_class) == topk:
                    break

        apk_score = ml_metrics.apk(
            [label_val[i]],
            pred_class, topk)
        apk_scores.append(apk_score)

    mapk_score = np.mean(apk_scores)
    return mapk_score


def search_with_faiss_cpu(feat_test, feat_index, topk=5):
    n_dim = feat_index.shape[1]
    with timer("Build index with faiss-cpu ...", logger=logger):
        cpu_index = faiss.IndexFlatIP(n_dim)
        cpu_index.add(feat_index)

    with timer("Search with faiss-cpu index ...", logger=logger):
        dists, topk_idx = cpu_index.search(x=feat_test, k=topk)

    cpu_index.reset()
    del cpu_index

    return dists, topk_idx


def search_with_faiss_gpu(feat_test, feat_index, topk=5):
    n_dim = feat_index.shape[1]
    with timer("Build index with faiss-gpu ...", logger=logger):
        cpu_index = faiss.IndexFlatIP(n_dim)
        cpu_index.add(feat_index)

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        vres = []
        for _ in range(faiss.get_num_gpus()):
            res = faiss.StandardGpuResources()
            vres.append(res)

        gpu_index = faiss.index_cpu_to_gpu_multiple_py(vres, cpu_index, co)

    with timer("Search with faiss-gpu index ...", logger=logger):
        dists, topk_idx = gpu_index.search(x=feat_test, k=topk)

    gpu_index.reset()
    cpu_index.reset()
    del cpu_index, gpu_index

    for res in vres:
        res.noTempMemory()
        del res

    return dists, topk_idx


def extract_features(model, dataloaders, gfc, target_list=["index", "val"]):
    for target in target_list:
        with timer(f"Extract {target} features ...", logger=logger):
            if gfc.contains(f"feat_{target}") and gfc.contains(f"label_{target}"):
                continue

            rows, label_rows = [], []
            n_batches = len(dataloaders[target])
            if n_batches > 100:
                logger.info(f"target set:{target}, num batches: {n_batches}")

            for idx, (X, y) in enumerate(dataloaders[target]):
                if (n_batches > 100) and (idx % 100 == 0):
                    logger.info(f" - processing {idx} of {n_batches}")

                # print("0", X.size(), X.dtype, model)
                X, y = X.to("cuda"), y.to("cuda")
                # print("1", X.size(), X.dtype, type(X), type(model))
                with torch.no_grad():
                    outputs = model.extract_features(X)
                # print("2")

                label_rows += y.data.cpu().numpy().tolist()
                batch_size = X.size(0)
                for j in range(batch_size):
                    rows.append(outputs[j].data.cpu().numpy().ravel())
                # print("3")

            feat = np.array(rows).astype(np.float32)
            feat = l2norm_numpy(feat)
            gfc.add(f"feat_{target}", feat)
            gfc.add(f"label_{target}", np.array(label_rows))


def evaluate_with_numpy(model, evaluation_set):
    for eval_dataset_name in evaluation_set.keys():
        with timer("Extract index features...", logger=logger):
            dataloaders = evaluation_set[eval_dataset_name]
            rows, label_rows = [], []
            for idx, (X, y) in enumerate(dataloaders["index"]):
                X, y = X.to("cuda"), y.to("cuda")
                with torch.no_grad():
                    outputs = model.extract_features(X)

                label_rows += y.data.cpu().numpy().tolist()
                batch_size = X.size(0)
                for j in range(batch_size):
                    rows.append(outputs[j].data.cpu().numpy().ravel())

            feat_index = np.array(rows).astype(np.float32)
            feat_index = l2norm_numpy(feat_index)

        print(feat_index.shape)
