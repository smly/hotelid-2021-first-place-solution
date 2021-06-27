import shutil
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm

import torch
import torch.nn.functional as F
from peko.augmentation import get_augv5_no_flip
from peko.cbir.functions import l2norm_numpy
from peko.cbir.search import extract_features
from peko.dataset.hotelid import (HotelDataset, HotelRotationFixedDataset,
                                  HotelRotationFixedTestDataset,
                                  HotelRotationFixedVNDataset,
                                  HotelTestDataset)
from peko.dataset.hotelid_rf_v2 import HotelV2RotationFixedDataset
from peko.torch.angular import AngularModel, AngularModelChainHead
from peko.utils.common import dynamic_load
from peko.utils.configs import load_config
from peko.utils.feature_container import GlobalFeatureContainer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

logger = getLogger("peko")


PATH_TRAIN_CSV = "/kaggle/input/hotel-id-2021-fgvc8/train.csv"
PATH_TRAIN_IMAGES = "/kaggle/input/hotel-id-2021-fgvc8/train_images"
PATH_TEST_IMAGES = "/kaggle/input/hotel-id-2021-fgvc8/test_images"

fn_train_hotelidv3 = "/kaggle/input/hotelid-models/train_hotelidv3.csv"
if not Path(fn_train_hotelidv3).exists():
    fn_train_hotelidv3 = "/kaggle/input/d/confirm/hotelid-models/train_hotelidv3.csv"

fn_train_hotelidv4 = "/kaggle/input/hotelid-models/train_hotelidv4.csv"
if not Path(fn_train_hotelidv4).exists():
    fn_train_hotelidv4 = "/kaggle/input/d/confirm/hotelid-models/train_hotelidv4.csv"

# Dirty workaround (what????? Kaggle bug?????)
hotelid_model_dir = "/kaggle/input/hotelid-models"
if not Path(hotelid_model_dir).exists():
    hotelid_model_dir = "/kaggle/input/d/confirm/hotelid-models"


def test_dataloader(
    filepath_list,
    test_batch_size=256,
    num_workers=8,
    test_transform=None,
    resize_shape=None,
):
    ds_test = HotelTestDataset(
        filepath_list,
        transform=test_transform,
        resize_shape=resize_shape,
    )
    return DataLoader(
        dataset=ds_test,
        sampler=SequentialSampler(ds_test),
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )


def test_RF_dataloader(
    filepath_list,
    rotation_fixed_path: str,
    test_batch_size=256,
    num_workers=8,
    test_transform=None,
    resize_shape=None,
):
    ds_test = HotelRotationFixedTestDataset(
        filepath_list,
        rotation_fixed_path,
        transform=test_transform,
        resize_shape=resize_shape,
    )
    return DataLoader(
        dataset=ds_test,
        sampler=SequentialSampler(ds_test),
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )


def index_dataloader(
    train_path: str,
    test_batch_size: int = 32,
    num_workers: int = 8,
    test_transform=None,
    resize_shape=None,
    img_dir=PATH_TRAIN_IMAGES,
):
    ds_trn = HotelDataset(
        train_path,
        transform=test_transform,
        resize_shape=resize_shape,
        img_dir=img_dir,
    )
    return DataLoader(
        dataset=ds_trn,
        sampler=SequentialSampler(ds_trn),
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


def index_RF_dataloader(
    train_path: str,
    rotation_fixed_path: str,
    test_batch_size: int = 32,
    num_workers: int = 8,
    test_transform=None,
    resize_shape=None,
    img_dir=PATH_TRAIN_IMAGES,
):
    ds_trn = HotelRotationFixedDataset(
        train_path,
        rotation_fixed_path,
        transform=test_transform,
        resize_shape=resize_shape,
        img_dir=img_dir,
    )
    return DataLoader(
        dataset=ds_trn,
        sampler=SequentialSampler(ds_trn),
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )


def index_RFvn_dataloader(
    train_path: str,
    test_batch_size: int = 32,
    num_workers: int = 8,
    test_transform=None,
    resize_shape=None,
    img_dir=PATH_TRAIN_IMAGES,
):
    ds_trn = HotelRotationFixedVNDataset(
        train_path,
        transform=test_transform,
        resize_shape=resize_shape,
        img_dir=img_dir,
    )
    return DataLoader(
        dataset=ds_trn,
        sampler=SequentialSampler(ds_trn),
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )


def index_RFv2_dataloader(
    train_path: str,
    rotation_fixed_path: str,
    test_batch_size: int = 32,
    num_workers: int = 8,
    test_transform=None,
    img_dir=PATH_TRAIN_IMAGES,
):
    """
    index_RF_dataloader は hotelid train set の dataloader.
    index_RFv2_dataloader は hotelidv2 train set の dataloader.
    """
    ds_trn = HotelV2RotationFixedDataset(
        train_path,
        rotation_fixed_path,
        transform=test_transform,
        img_dir=img_dir,
    )
    return DataLoader(
        dataset=ds_trn,
        sampler=SequentialSampler(ds_trn),
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )


class HotelIDInferRotationDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        assert Path(path).exists()
        im = cv2.imread(str(path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            im = self.transform(image=im)["image"]
        im = torch.from_numpy(im.transpose((2, 0, 1))).float()

        return im


def accuracy(true, pred):
    acc = (true == pred.argmax(-1)).float().detach().cpu().numpy()
    return float(acc.sum() / len(acc))


def load_rf_models(model_weight_dict, rot_csv, method=None):
    testdataset, label_index = None, None
    feat_test_list = []
    feat_index_list = []
    model_names = model_weight_dict.keys()
    for i, model_name in enumerate(model_names):
        logger.info(f"- model: {model_name}")
        # conf, model, test_dataloaders, gfc
        cfg_path = f"{hotelid_model_dir}/{model_name}.yml"

        # Copy GFC
        if method is None:
            epoch = 10
            feature_path = str(Path(
                hotelid_model_dir
            ) / f"{model_name}_epoch{epoch:d}_RF_index.h5")
            logger.info(f"- load index: {feature_path}")
            shutil.copy(feature_path, "/tmp/")
        elif method == "v3":
            epoch = 10
            feature_path = str(Path(
                hotelid_model_dir
            ) / f"{model_name}_epoch{epoch:d}_RFv3_index.h5")
            logger.info(f"- load index: {feature_path}")
            shutil.copy(feature_path, "/tmp/")
        elif method == "v4":
            epoch = 10
            feature_path = str(Path(
                hotelid_model_dir
            ) / f"{model_name}_epoch{epoch:d}_RFv4_index.h5")
            logger.info(f"- load index: {feature_path}")
            shutil.copy(feature_path, "/tmp/")
        else:
            raise RuntimeError

        if method is None:
            rets = load_all_test_with_rotation_fix(
                cfg_path,
                rot_csv,
                test_batch_size=16,
                num_workers=2,
                storage_dir="/tmp")
        elif method == "v3":
            rets = load_all_test_with_rotation_fix_v3(
                cfg_path,
                rot_csv,
                test_batch_size=16,
                num_workers=2,
                storage_dir="/tmp")
        elif method == "v4":
            rets = load_all_test_with_rotation_fix_v4(
                cfg_path,
                rot_csv,
                test_batch_size=16,
                num_workers=2,
                storage_dir="/tmp")
        else:
            raise RuntimeError

        extract_features(rets[1], rets[2], rets[3], target_list=["test"])

        if i == 0:
            testdataset = rets[2]["test"].dataset
            label_index = rets[3]["label_index"]

        if len(model_names) > 1:
            feat_test_list.append(rets[3]["feat_test"] * model_weight_dict[model_name])
            feat_index_list.append(rets[3]["feat_index"] * model_weight_dict[model_name])
        else:
            feat_test_list.append(rets[3]["feat_test"])
            feat_index_list.append(rets[3]["feat_index"])

    if len(feat_test_list) > 1:
        feat_test = np.hstack(feat_test_list)
        feat_index = np.hstack(feat_index_list)
        del feat_test_list, feat_index_list
        feat_test = l2norm_numpy(feat_test)
        feat_index = l2norm_numpy(feat_index)
    else:
        feat_test = feat_test_list[0]
        feat_index = feat_index_list[0]

    return testdataset, feat_test, feat_index, label_index


def make_hidev2_sub(preds, test_ds):
    logger.info("WARNING: 1/10 predictions are all zero.")

    with open("submission.csv", "w") as f:
        f.write("image,hotel_id\n")
        for i, pred in enumerate(preds):
            if i % 10 == 0:
                filename = test_ds.paths[i].split("/")[-1]
                f.write(f"{filename},0 0 0 0 0\n")
            else:
                filename = test_ds.paths[i].split("/")[-1]
                pred_str = " ".join([str(class_id) for class_id in pred])
                f.write(f"{filename},{pred_str}\n")


def make_hide_sub(preds, test_ds):
    logger.info("WARNING: 1/3 predictions are including top1 dummy predictions.")

    with open("submission.csv", "w") as f:
        f.write("image,hotel_id\n")
        for i, pred in enumerate(preds):
            if i % 3 == 0:
                filename = test_ds.paths[i].split("/")[-1]
                pred_str = " ".join([str(class_id) for class_id in pred][:4])
                f.write(f"{filename},0 {pred_str}\n")
            else:
                filename = test_ds.paths[i].split("/")[-1]
                pred_str = " ".join([str(class_id) for class_id in pred])
                f.write(f"{filename},{pred_str}\n")


def make_normal_sub(preds, test_ds):
    with open("submission.csv", "w") as f:
        f.write("image,hotel_id\n")
        for i, pred in enumerate(preds):
            filename = test_ds.paths[i].split("/")[-1]
            pred_str = " ".join([str(class_id) for class_id in pred])
            f.write(f"{filename},{pred_str}\n")


def infer_test_rotations_with_v80model(output_csv):
    weight_path = f"{hotelid_model_dir}/v80_rotation_model_epoch08.pth"
    infer_test_rotations(
        weight_path,
        list(sorted(Path(
            "/kaggle/input/hotel-id-2021-fgvc8/test_images"
        ).glob("*.jpg"))), output_csv)


def infer_test_rotations(weight_path, eval_paths, output_csv):
    # output_csv = "v80_rotation_hotelid_test_images.csv"
    # weight_path = "data/working/models/v80_rotation_model/"
    # weight_path += "v80_rotation_model_epoch08.pth"
    # eval_paths = list(sorted([
    #     str(p)
    #     for p in Path("data/input/train_images/").glob("*/*.jpg")
    # ]))

    backbone_name = "resnet50"
    model = timm.create_model(backbone_name, pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()
    model.eval()

    batch_size = 32
    input_size = 256
    _, tst_transform = get_augv5_no_flip(input_size)
    tst_ds = HotelIDInferRotationDataset(eval_paths, transform=tst_transform)
    tst_dl = DataLoader(
        tst_ds,
        sampler=SequentialSampler(tst_ds),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )

    ret = []
    ds_idx = 0
    for idx, X in enumerate(tst_dl):
        X = X.cuda()
        with torch.no_grad():
            out = model(X)
            prob = F.softmax(out, dim=1)
            rot = prob.argmax(dim=1).long().cpu().detach().numpy()
            conf_score = torch.max(prob, dim=1)[0].float().cpu().detach().numpy()
            bs = out.size(0)
            for i in range(bs):
                ret.append({
                    "path": tst_ds.paths[ds_idx],
                    "rot": rot[i],
                    "score": conf_score[i],
                })
                ds_idx += 1

    pd.DataFrame(ret)[[
        "path",
        "rot",
        "score",
    ]].to_csv(output_csv, index=False)


def load_all_test_stuff(
    config_path,
    storage_dir="./",
    test_batch_size=32,
    num_workers=4,
    epoch=None
):
    conf = load_config(config_path)

    # Dataloaders
    get_aug = dynamic_load(conf["augmentation_func"])
    _, test_transform = get_aug(conf["input_size"])

    filepath_list = list(sorted([
        str(p) for p in Path(PATH_TEST_IMAGES).glob("*.jpg")
    ]))

    dataloaders = {}
    dataloaders["index"] = index_dataloader(
        PATH_TRAIN_CSV,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )
    dataloaders["test"] = test_dataloader(
        filepath_list,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )

    epoch = epoch or conf.total_epochs
    feature_path = str(Path(storage_dir) / f"{conf.name}_epoch{epoch:d}_index.h5")
    gfc = GlobalFeatureContainer(feature_path)

    weight_path = f"{hotelid_model_dir}/{conf.name}_epoch{epoch:d}.pth"
    if "model" in conf and conf.get("api_version", 1) == 1:
        model_kwargs = conf.model.kwargs.copy()
        model_kwargs["pretrained"] = False
        model_cls = dynamic_load(conf.model.fqdn)
        model = model_cls(**model_kwargs)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = AngularModel(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()

    return conf, model, dataloaders, gfc


def load_all_test_with_rotation_fix_v4(
    config_path,
    rotation_fixed_filepath,
    storage_dir="./",
    test_batch_size=32,
    num_workers=4,
    epoch=None,
    resolution=None,
):
    conf = load_config(config_path)
    input_size = conf["input_size"]
    if resolution:
        input_size = resolution

    # Dataloaders
    get_aug = dynamic_load(conf["augmentation_func"])
    _, test_transform = get_aug(input_size)

    filepath_list = list(sorted([
        str(p) for p in Path(PATH_TEST_IMAGES).glob("*.jpg")
    ]))

    dataloaders = {}
    dataloaders["index"] = index_RFvn_dataloader(
        fn_train_hotelidv4,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )
    dataloaders["test"] = test_RF_dataloader(
        filepath_list,
        rotation_fixed_filepath,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )

    epoch = epoch or conf.total_epochs
    if resolution:
        feature_path = str(Path(
            storage_dir) / f"{conf.name}_epoch{epoch:d}_r{input_size}_RFv4_index.h5")
    else:
        feature_path = str(Path(storage_dir) / f"{conf.name}_epoch{epoch:d}_RFv4_index.h5")

    gfc = GlobalFeatureContainer(feature_path)

    weight_path = f"{hotelid_model_dir}/{conf.name}_epoch{epoch:d}.pth"
    if "model" in conf and conf.get("api_version", 1) == 1:
        model_kwargs = conf.model.kwargs.copy()
        model_kwargs["pretrained"] = False
        model_cls = dynamic_load(conf.model.fqdn)
        model = model_cls(**model_kwargs)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = AngularModel(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()

    return conf, model, dataloaders, gfc


def load_all_test_with_rotation_fix_v3(
    config_path,
    rotation_fixed_filepath,
    storage_dir="./",
    test_batch_size=32,
    num_workers=4,
    epoch=None,
    resolution=None,
):
    conf = load_config(config_path)
    input_size = conf["input_size"]
    if resolution:
        input_size = resolution

    # Dataloaders
    get_aug = dynamic_load(conf["augmentation_func"])
    _, test_transform = get_aug(input_size)

    filepath_list = list(sorted([
        str(p) for p in Path(PATH_TEST_IMAGES).glob("*.jpg")
    ]))

    dataloaders = {}
    dataloaders["index"] = index_RFvn_dataloader(
        fn_train_hotelidv3,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )
    dataloaders["test"] = test_RF_dataloader(
        filepath_list,
        rotation_fixed_filepath,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )

    epoch = epoch or conf.total_epochs
    if resolution:
        feature_path = str(Path(
            storage_dir) / f"{conf.name}_epoch{epoch:d}_r{input_size}_RFv3_index.h5")
    else:
        feature_path = str(Path(storage_dir) / f"{conf.name}_epoch{epoch:d}_RFv3_index.h5")

    gfc = GlobalFeatureContainer(feature_path)

    weight_path = f"{hotelid_model_dir}/{conf.name}_epoch{epoch:d}.pth"
    if "model" in conf and conf.get("api_version", 1) == 1:
        model_kwargs = conf.model.kwargs.copy()
        model_kwargs["pretrained"] = False
        model_cls = dynamic_load(conf.model.fqdn)
        model = model_cls(**model_kwargs)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = AngularModel(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()

    return conf, model, dataloaders, gfc


def load_all_test_with_rotation_fix(
    config_path,
    rotation_fixed_filepath,
    storage_dir="./",
    test_batch_size=32,
    num_workers=4,
    epoch=None,
    resolution=None,
):
    conf = load_config(config_path)
    input_size = conf["input_size"]
    if resolution:
        input_size = resolution

    # Dataloaders
    get_aug = dynamic_load(conf["augmentation_func"])
    _, test_transform = get_aug(input_size)

    filepath_list = list(sorted([
        str(p) for p in Path(PATH_TEST_IMAGES).glob("*.jpg")
    ]))

    dataloaders = {}
    dataloaders["index"] = index_RF_dataloader(
        PATH_TRAIN_CSV,
        None,  # dummy
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )
    dataloaders["test"] = test_RF_dataloader(
        filepath_list,
        rotation_fixed_filepath,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )

    epoch = epoch or conf.total_epochs
    if resolution:
        feature_path = str(Path(
            storage_dir) / f"{conf.name}_epoch{epoch:d}_r{input_size}_RF_index.h5")
    else:
        feature_path = str(Path(storage_dir) / f"{conf.name}_epoch{epoch:d}_RF_index.h5")

    gfc = GlobalFeatureContainer(feature_path)

    weight_path = f"{hotelid_model_dir}/{conf.name}_epoch{epoch:d}.pth"
    if "model" in conf and conf.get("api_version", 1) == 1:
        model_kwargs = conf.model.kwargs.copy()
        model_kwargs["pretrained"] = False
        model_cls = dynamic_load(conf.model.fqdn)
        model = model_cls(**model_kwargs)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = AngularModel(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()

    return conf, model, dataloaders, gfc


def load_all_test_with_rotation_fix_v2(
    config_path,
    rotation_fixed_filepath,
    storage_dir="./",
    test_batch_size=32,
    num_workers=4,
    epoch=None,
    resolution=None,
):
    conf = load_config(config_path)
    input_size = conf["input_size"]
    if resolution:
        input_size = resolution

    # Dataloaders
    get_aug = dynamic_load(conf["augmentation_func"])
    _, test_transform = get_aug(input_size)

    filepath_list = list(sorted([
        str(p) for p in Path(PATH_TEST_IMAGES).glob("*.jpg")
    ]))

    dataloaders = {}
    dataloaders["test"] = test_RF_dataloader(
        filepath_list,
        rotation_fixed_filepath,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_transform=test_transform,
        resize_shape=None,
    )

    epoch = epoch or conf.total_epochs
    if resolution:
        feature_path = str(Path(
            storage_dir) / f"{conf.name}_epoch{epoch:d}_r{input_size}_RFv2_index.h5")
    else:
        feature_path = str(Path(storage_dir) / f"{conf.name}_epoch{epoch:d}_RFv2_index.h5")

    gfc = GlobalFeatureContainer(feature_path)

    weight_path = f"{hotelid_model_dir}/{conf.name}_epoch{epoch:d}.pth"
    if "model" in conf and conf.get("api_version", 1) == 1:
        model_kwargs = conf.model.kwargs.copy()
        model_kwargs["pretrained"] = False
        model_cls = dynamic_load(conf.model.fqdn)
        model = model_cls(**model_kwargs)
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = AngularModel(
            n_classes=conf["n_classes"], model_name=conf["backbone"], pretrained=False)
        model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()

    return conf, model, dataloaders, gfc
