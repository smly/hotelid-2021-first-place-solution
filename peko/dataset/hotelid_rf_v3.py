from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler


class HotelV3RotationFixedDataset(Dataset):
    def __init__(self,
                 dataframe_filepath="data/working/feats/v110_hidv2_hotelidv2.csv",
                 rotation_fixed_filepath="data/working/v80_rotation_hotelidv3_train_images.csv",
                 transform=None,
                 size_template=None,
                 img_dir="data/input/train_images"):
        self.paths = []
        self.class_ids = []
        self.img_dir = img_dir

        self.hotelid_idx_map = {}
        self.__load_path_label(dataframe_filepath)
        self.transform = transform

        self.rotation_dict = self.__load_rotation_db(rotation_fixed_filepath)

    def __load_rotation_db(self, path):
        rotation_dict = {}
        if path is not None:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                filename = Path(r["path"]).name
                if r["score"] > 0.9 and r["rot"] > 0:
                    rotation_dict[filename] = r["rot"]
        return rotation_dict

    def __load_path_label(self, path):
        img_dir = self.img_dir

        df = pd.read_csv(path, low_memory=False)

        # Initialize `hotelid_idx_map`
        hotel_ids = list(sorted(df["hotel_id"].unique()))
        self.hotelid_idx_map = dict(zip(hotel_ids, list(range(len(hotel_ids)))))

        for _, r in df.iterrows():
            image, chain = r["image"], r["chain"]
            if r["source"] == "hotels50k":
                if Path(r["image"]).exists():
                    self.paths.append(r["image"])
                    self.class_ids.append(self.hotelid_idx_map[r["hotel_id"]])
            else:
                path = f"{img_dir}/{int(chain)}/{image}"
                self.paths.append(path)
                self.class_ids.append(self.hotelid_idx_map[r["hotel_id"]])

    def __getitem__(self, index):
        path = self.paths[index]
        assert Path(path).exists()
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        filename = Path(path).name
        rot = self.rotation_dict.get(filename, 0)
        if rot > 0:
            if rot == 1:
                im = np.rot90(im, 3)
            elif rot == 2:
                im = np.rot90(im, 2)
            elif rot == 3:
                im = np.rot90(im, 1)

        assert im is not None
        im = im[..., ::-1]

        if self.transform:
            im = self.transform(image=im)["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        target = torch.tensor(self.class_ids[index]).long()
        return im, target

    def __len__(self):
        return len(self.paths)


class HotelidV3(Dataset):
    def __init__(self,
                 dataframe_filepath="data/working/feats/v110_hidv2_hotelidv2.csv",
                 transform=None,
                 size_template=None,
                 img_dir="data/input/train_images"):
        self.paths = []
        self.class_ids = []
        self.img_dir = img_dir

        self.hotelid_idx_map = {}
        self.__load_path_label(dataframe_filepath)
        self.transform = transform

        rotation_fixed_filepath = "data/working/v80_rotation_hotelidv3_train_images.csv"
        self.rotation_dict = self.__load_rotation_db(rotation_fixed_filepath)

    def __load_rotation_db(self, path):
        rotation_dict = {}
        if path is not None:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                filename = Path(r["path"]).name
                if r["score"] > 0.9 and r["rot"] > 0:
                    rotation_dict[filename] = r["rot"]
        return rotation_dict

    def __load_path_label(self, path):
        img_dir = self.img_dir

        df = pd.read_csv(path, low_memory=False)

        # Initialize `hotelid_idx_map`
        hotel_ids = list(sorted(df["hotel_id"].unique()))
        self.hotelid_idx_map = dict(zip(hotel_ids, list(range(len(hotel_ids)))))

        for _, r in df.iterrows():
            image, chain = r["image"], r["chain"]
            if r["source"] == "hotels50k":
                if Path(r["image"]).exists():
                    self.paths.append(r["image"])
                    self.class_ids.append(self.hotelid_idx_map[r["hotel_id"]])
            else:
                path = f"{img_dir}/{int(chain)}/{image}"
                self.paths.append(path)
                self.class_ids.append(self.hotelid_idx_map[r["hotel_id"]])

    def __getitem__(self, index):
        path = self.paths[index]
        assert Path(path).exists()
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        filename = Path(path).name
        rot = self.rotation_dict.get(filename, 0)
        if rot > 0:
            if rot == 1:
                im = np.rot90(im, 3)
            elif rot == 2:
                im = np.rot90(im, 2)
            elif rot == 3:
                im = np.rot90(im, 1)

        assert im is not None
        im = im[..., ::-1]

        if self.transform:
            im = self.transform(image=im)["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        target = torch.tensor(self.class_ids[index]).long()
        return im, target

    def __len__(self):
        return len(self.paths)


def train_dataloaders(
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    num_workers: int = 8,
    train_transform=None,
    val_transform=None,
    resize_shape=(320, 240),
    is_distributed=False,
    img_dir="data/input/train_images",
):
    ds_trn = HotelidV3(transform=train_transform, img_dir=img_dir)
    dataloaders = {}
    dataloaders["train"] = DataLoader(
        dataset=ds_trn,
        sampler=RandomSampler(ds_trn),
        batch_size=train_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataloaders


def index_dataloader(
    index_batch_size: int = 32,
    num_workers: int = 8,
    index_transform=None,
    resize_shape=(320, 240),
    is_distributed=False,
    img_dir="data/input/train_images",
):
    ds_trn = HotelidV3(transform=index_transform, img_dir=img_dir)
    return DataLoader(
        dataset=ds_trn,
        sampler=RandomSampler(ds_trn),
        batch_size=index_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


def get_dataloaders(train_transform, test_transform, params):
    dataloaders = train_dataloaders(
        train_batch_size=params["train_batch_size"],
        val_batch_size=params["val_batch_size"],
        num_workers=params["num_workers"],
        train_transform=train_transform,
        val_transform=test_transform,
    )
    dl_index = index_dataloader(
        index_batch_size=params["val_batch_size"],
        num_workers=params["num_workers"],
        index_transform=test_transform,
    )
    dataloaders["index"] = dl_index

    return dataloaders
