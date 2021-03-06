import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from logging import getLogger
from pathlib import Path

import cv2
import pandas as pd
import torch
from peko.cbir.search import extract_features
from peko.competition import index_RF_dataloader, index_RFvn_dataloader
from peko.torch.angular import AngularModel
from peko.utils.common import dynamic_load
from peko.utils.configs import load_config
from peko.utils.feature_container import GlobalFeatureContainer
from peko.utils.logger import set_logger

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger = getLogger("peko")

FN_TRAIN_HOTELIDV3 = "data/working/train_hotelidv3.csv"


def prepare_dataset():
    """
    train.csv + traffickcam images from h50k を作成する

    train_hotelidv3.csv
    image,chain,hotel_id,timestamp
    """
    FN_TRAIN_EXT = "data/working/feats/v110_hidv2_hotelidv2.csv"
    FN_ROT = "data/working/v80_rotation_hotelidv3_train_images.csv"
    if Path(FN_TRAIN_HOTELIDV3).exists():
        return

    rot_dict = {}
    df_rot = pd.read_csv(FN_ROT)
    for _, r in df_rot.iterrows():
        rot_dict[r["path"]] = (r["rot"], r["score"])

    rows = []
    df = pd.read_csv(FN_TRAIN_EXT)
    for _, r in df.iterrows():
        if r["source"] == "hotel-id":
            chain_id = int(r["chain"])
            hotel_id = int(r["hotel_id"])
            filename = r["image"]
            path = f"data/input/train_images/{chain_id}/{filename}"
            assert Path(path).exists()
            rows.append({
                "image": path,
                "chain": chain_id,
                "hotel_id": hotel_id,
                "rot": rot_dict[path][0],
                "rot_score": rot_dict[path][1],
            })
        else:
            filepath = r["image"]
            chain_id = 0
            hotel_id = int(r["hotel_id"])
            if "travel_website" in filepath:
                continue
            if Path(filepath).exists():
                rows.append({
                    "image": filepath,
                    "chain": chain_id,
                    "hotel_id": hotel_id,
                    "rot": rot_dict[filepath][0],
                    "rot_score": rot_dict[filepath][1],
                })
    df = pd.DataFrame(rows)
    df[["image", "chain", "hotel_id", "rot", "rot_score"]].to_csv(
        FN_TRAIN_HOTELIDV3,
        index=False
    )


def main(args: argparse.Namespace):
    prepare_dataset()

    conf = load_config(args.config)
    epoch = args.epoch
    set_logger(logger)

    input_size = conf["input_size"]
    if args.resolution:
        input_size = args.resolution

    # Dataloaders
    get_aug = dynamic_load(conf["augmentation_func"])
    _, test_transform = get_aug(input_size)

    dataloaders = {}
    dataloaders["index"] = index_RFvn_dataloader(
        FN_TRAIN_HOTELIDV3,
        test_batch_size=args.batch_size or 128,
        num_workers=args.worker_size or 8,
        test_transform=test_transform,
        resize_shape=None,
        img_dir="data/input/train_images",
    )

    epoch = epoch or conf.total_epochs
    storage_dir = "data/working/feats/"
    feature_path = str(Path(storage_dir) / f"{conf.name}_epoch{epoch:d}_RFv3_index.h5")
    if args.resolution:
        feature_path = str(Path(storage_dir) / (
            f"{conf.name}_epoch{epoch:d}_r{input_size}_RFv3_index.h5"))

    gfc = GlobalFeatureContainer(feature_path)

    model_dir = "data/working/models/"
    weight_path = str(Path(model_dir) / conf.name / f"{conf.name}_epoch{epoch:d}.pth")
    if "model" in conf and conf.get("api_version", 1) == 1:
        model_kwargs = conf.model.kwargs
        model_cls = dynamic_load(conf.model.fqdn)
        model = model_cls(**model_kwargs)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = AngularModel(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()

    extract_features(model, dataloaders, gfc, target_list=["index"])


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)

    # ResNeSt101e, 768x768, RTX3090 の場合は bs=64
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--worker-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
