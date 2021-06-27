import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from peko.cbir.functions import l2norm_numpy
from peko.cbir.search import extract_features, search_with_faiss_cpu
from peko.competition import index_RFvn_dataloader
from peko.torch.angular import AngularModel
from peko.utils.common import dynamic_load
from peko.utils.configs import load_config
from peko.utils.feature_container import GlobalFeatureContainer
from peko.utils.logger import set_logger

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger = getLogger("peko")

FN_TRAIN_V2_HOTELS50K_NO_ROTATION = "data/working/feats/v110_hidv2_hotelidv2.csv"
FN_TRAIN_V2_HOTELS50K_CANDIDATES = "data/working/train_hotelidv4_candidates_hotels50k.csv"
FN_TRAIN_HOTELIDV3 = "data/working/train_hotelidv3.csv"
FN_TRAIN_HOTELIDV4_HOTELS50K_INCLUDED = "data/working/train_hotelidv4_h50k_included.csv"
FN_TRAIN_HOTELIDV4 = "data/working/train_hotelidv4.csv"

# RTv4 ... travel_site 画像から traffickcam へ検索して best 1 が同じラベルであれば追加する


def generate_part_of_v4_hotels50k_data():
    if Path(FN_TRAIN_V2_HOTELS50K_CANDIDATES).exists():
        return

    df = pd.read_csv(FN_TRAIN_V2_HOTELS50K_NO_ROTATION)
    df = df[(df["source"] == "hotels50k") & (df["image"].str.contains("travel_website"))][[
        "image",
        "hotel_id",
    ]]
    rows = []
    for _, r in df.iterrows():
        if Path(r["image"]).exists():
            rows.append({
                "image": r["image"],
                "hotel_id": r["hotel_id"],
                "rot": 0,
                "rot_score": 1.0,
            })
    pd.DataFrame(rows)[[
        "image",
        "hotel_id",
        "rot",
        "rot_score",
    ]].to_csv(FN_TRAIN_V2_HOTELS50K_CANDIDATES, index=False)


def load_test():
    conf = load_config("configs/v131_regnety_augv5_s512_hotelidv3.yml")
    epoch = 10
    batch_size = 92
    worker_size = 8
    set_logger(logger)

    input_size = conf["input_size"]

    # Dataloaders
    get_aug = dynamic_load(conf["augmentation_func"])
    _, test_transform = get_aug(input_size)

    dl = index_RFvn_dataloader(
        FN_TRAIN_V2_HOTELS50K_CANDIDATES,
        test_batch_size=batch_size,
        num_workers=worker_size,
        test_transform=test_transform,
        resize_shape=None,
        img_dir="data/input/train_images",
    )
    gfc = GlobalFeatureContainer("data/working/feats/hotelids_v4_candidates.h5")

    epoch = epoch or conf.total_epochs
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
    model.eval()

    feat_ext = []
    for idx, (X, _) in enumerate(dl):
        if idx % 10 == 0:
            print(idx, len(dl))
        with torch.no_grad():
            X = X.to("cuda")
            out = model.extract_features(X)
            feat_ext.append(out.data.cpu().numpy())
    feat_ext = np.vstack(feat_ext)
    feat_ext = l2norm_numpy(feat_ext)
    feat_ext = feat_ext.astype(np.float32)
    gfc.add("feat_ext", feat_ext)
    print(feat_ext.shape)


def main2():
    gfc = GlobalFeatureContainer("data/working/feats/hotelids_v4_candidates.h5")
    feat_ext = np.array(gfc["feat_ext"])
    df_ext = pd.read_csv(FN_TRAIN_V2_HOTELS50K_CANDIDATES)
    label_ext = np.array(df_ext["hotel_id"])
    del gfc

    df_index = pd.read_csv("data/working/train_hotelidv3.csv")
    fn_index = "data/working/feats/v131_regnety_augv5_s512_hotelidv3_epoch10_RFv3_index.h5"
    gfc = GlobalFeatureContainer(fn_index)
    feat_index = np.array(gfc["feat_index"])
    label_index = np.array(gfc["label_index"])
    del gfc

    print(len(df_ext), feat_ext.shape, label_ext.shape, feat_index.shape, label_index.shape)

    dists, topk_idxs = search_with_faiss_cpu(feat_ext, feat_index, topk=5)
    rows = []
    for i in range(dists.shape[0]):
        rows.append({
            "hotels50k": df_ext.iloc[i]["image"],
            "traffickcam": df_index.iloc[topk_idxs[i, 0]]["image"],
            "hotels50k_label": label_ext[i],
            "hotelid_label": label_index[topk_idxs[i, 0]],
            "matched": 1 if label_ext[i] == label_index[topk_idxs[i, 0]] else 0,
        })
    print(len(rows), pd.DataFrame(rows)["matched"].sum())
    pd.DataFrame(rows).to_csv(FN_TRAIN_HOTELIDV4_HOTELS50K_INCLUDED, index=False)


def generate_v4():
    df_h50k = pd.read_csv(FN_TRAIN_HOTELIDV4_HOTELS50K_INCLUDED)
    df_hidv3 = pd.read_csv(FN_TRAIN_HOTELIDV3)
    df_h50k = df_h50k[df_h50k["matched"] > 0]

    rows = []
    for _, r in df_h50k.iterrows():
        rows.append({
            "image": r["hotels50k"],
            "chain": 0,
            "hotel_id": r["hotels50k_label"],
            "rot": 0,
            "rot_score": 1.0,
        })

    df_hidv4 = pd.concat([df_hidv3, pd.DataFrame(rows)])
    df_hidv4[[
        "image",
        "chain",
        "hotel_id",
        "rot",
        "rot_score",
    ]].to_csv(FN_TRAIN_HOTELIDV4, index=False)


def main(args: argparse.Namespace):
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
        FN_TRAIN_HOTELIDV4,
        test_batch_size=args.batch_size or 128,
        num_workers=args.worker_size or 8,
        test_transform=test_transform,
        resize_shape=None,
        img_dir="data/input/train_images",
    )

    epoch = epoch or conf.total_epochs
    storage_dir = "data/working/feats/"
    feature_path = str(Path(storage_dir) / f"{conf.name}_epoch{epoch:d}_RFv4_index.h5")
    if args.resolution:
        feature_path = str(Path(storage_dir) / (
            f"{conf.name}_epoch{epoch:d}_r{input_size}_RFv4_index.h5"))

    gfc = GlobalFeatureContainer(feature_path)

    model_dir = "data/working/models/"
    weight_path = str(Path(model_dir) / conf.name / f"{conf.name}_epoch{epoch:d}.pth")
    if conf.get("api_version", 1) == 5:
        model = AngularModel2(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt["state_dict"])
    elif conf.get("api_version", 1) == 6:
        raise RuntimeError
    elif "model" in conf and conf.get("api_version", 1) == 1:
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
    # generate_part_of_v4_hotels50k_data()
    # load_test()
    # main2()
    # generate_v4()
    main(parse_args())
