import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from logging import getLogger
from pathlib import Path

import cv2
import torch
from peko.cbir.search import extract_features
from peko.competition import index_dataloader
from peko.torch.angular import AngularModel, AngularModelChainHead
from peko.utils.common import dynamic_load
from peko.utils.configs import load_config
from peko.utils.feature_container import GlobalFeatureContainer
from peko.utils.logger import set_logger

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger = getLogger("peko")


def main(args: argparse.Namespace):
    conf = load_config(args.config)
    epoch = args.epoch
    set_logger(logger)

    # Dataloaders
    get_aug = dynamic_load(conf["augmentation_func"])
    _, test_transform = get_aug(conf["input_size"])

    dataloaders = {}
    dataloaders["index"] = index_dataloader(
        "data/input/train.csv",
        test_batch_size=args.batch_size or 128,
        num_workers=args.worker_size or 8,
        test_transform=test_transform,
        resize_shape=None,
        img_dir="data/input/train_images",
    )

    epoch = epoch or conf.total_epochs
    storage_dir = "data/working/feats/"
    feature_path = str(Path(storage_dir) / f"{conf.name}_epoch{epoch:d}_index.h5")
    gfc = GlobalFeatureContainer(feature_path)

    model_dir = "data/working/models/"
    weight_path = str(Path(model_dir) / conf.name / f"{conf.name}_epoch{epoch:d}.pth")

    if conf.get("api_version", 1) == 1:
        model = AngularModel(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        model.load_state_dict(torch.load(weight_path)["state_dict"])
    else:
        raise RuntimeError

    model = model.cuda()

    extract_features(model, dataloaders, gfc, target_list=["index"])


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=None)

    # ResNeSt101e, 768x768, RTX3090 の場合は bs=64
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--worker-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
