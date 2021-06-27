import argparse
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from peko.augmentation import get_augv4 as get_aug
from peko.cbir.dba import qe_dba
from peko.cbir.search import (compute_mapk_score, extract_features,
                              search_with_faiss_cpu)
from peko.dataset.factory import get_dataloaders
from peko.path import get_last_model_path, get_logpath
from peko.torch.angular import AngularModel
from peko.utils.configs import load_config
from peko.utils.feature_container import GlobalFeatureContainer
from peko.utils.logger import add_log_filehandler, set_logger
from peko.utils.timer import timer

logger = getLogger("peko")


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--quick", action="store_true", default=False)
    parser.add_argument("--unique", action="store_true", default=False)
    return parser.parse_args()


def main(args: argparse.Namespace):
    conf = load_config(args.config)

    logfile_path, _ = get_logpath(conf.name, suffix="evaluate")
    set_logger(logger)
    add_log_filehandler(logger, conf.name, logfile_path)
    logger.info(conf.name + " :=\n" + OmegaConf.to_yaml(conf))

    if args.quick:
        search_only(args, conf)
        return

    evaluation_set = {}
    with timer("Build dataloaders ... ", logger=logger):
        for eval_dataset_name in conf.eval_dataset:
            logger.info(f" - Processing for {eval_dataset_name}")
            train_transform, val_transform = get_aug(conf["input_size"])
            dataloaders = get_dataloaders(
                eval_dataset_name, train_transform, val_transform, conf)
            evaluation_set[eval_dataset_name] = dataloaders

    with timer("Load model weights ...", logger=logger):
        weight_path = get_last_model_path(conf, epoch=args.epoch)
        model = AngularModel(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        model.load_state_dict(torch.load(weight_path)["state_dict"])
        model = model.cuda()

    # Extract features
    for eval_dataset_name in conf.eval_dataset:
        feature_dirname = Path(weight_path).stem
        feature_path = Path("data/working/feats") / feature_dirname / f"{eval_dataset_name}.h5"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        if not feature_path.exists():
            gfc = GlobalFeatureContainer(feature_path)
            # Extract features from dataloaders["index"] and dataloaders["val"]
            extract_features(
                model,
                evaluation_set[eval_dataset_name],
                gfc,
                target_list=["index", "val"])

        feat_index = gfc.get("feat_index")
        feat_test = gfc.get("feat_val")
        dists, topk_idx = search_with_faiss_cpu(feat_test, feat_index, topk=100)
        mapk_score = compute_mapk_score(gfc, dists, topk_idx, query="val", index="index")
        logger.info(f"{feature_dirname}/{eval_dataset_name} MAP@5: {mapk_score:.6f}")


def search_only(args, conf):
    weight_path = get_last_model_path(conf, epoch=args.epoch)

    for eval_dataset_name in conf.eval_dataset:
        feature_dirname = Path(weight_path).stem
        feature_path = Path("data/working/feats") / feature_dirname / f"{eval_dataset_name}.h5"
        gfc = GlobalFeatureContainer(feature_path)

        feat_index, feat_test = gfc.get("feat_index"), gfc.get("feat_val")
        label_index, label_test = gfc.get("label_index"), gfc.get("label_val")

        # Predict unique labels.
        if args.unique:
            dists, topk_idx = search_with_faiss_cpu(feat_test, feat_index, topk=100)
            mapk_score = compute_mapk_score(
                label_test, label_index, dists, topk_idx, topk=5, unique=True)
        else:
            dists, topk_idx = search_with_faiss_cpu(feat_test, feat_index, topk=5)
            mapk_score = compute_mapk_score(
                label_test, label_index, dists, topk_idx, topk=5, unique=False)

        logger.info(f"{feature_dirname}/{eval_dataset_name} MAP@5: {mapk_score:.6f}")

        # DBA
        if True:
            n_qe, alpha = 5, 3.0
            feat_all = np.concatenate([feat_test, feat_index], axis=0)
            dists, topk_idx = search_with_faiss_cpu(feat_all, feat_all, topk=n_qe)
            feat_test, feat_index = qe_dba(
                feat_test, feat_index, dists, topk_idx, alpha=alpha, qe=False)

            dists, topk_idx = search_with_faiss_cpu(feat_test, feat_index, topk=5)
            mapk_score = compute_mapk_score(label_test, label_index, dists, topk_idx, topk=5)
            logger.info(f"{feature_dirname}/{eval_dataset_name} MAP@5: {mapk_score:.6f}")


if __name__ == "__main__":
    main(parse_args())
