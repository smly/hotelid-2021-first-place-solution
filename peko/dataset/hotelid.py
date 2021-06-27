from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class HotelRotationFixedTestDataset(Dataset):
    def __init__(self,
                 filepath_list,
                 rotation_fixed_filepath,
                 transform=None,
                 size_template=None,
                 resize_shape=(320, 240)):
        self.paths = filepath_list
        self.transform = transform
        self.resize_shape = resize_shape
        self.rotation_dict = self.__load_rotation_db(rotation_fixed_filepath)

    def __load_rotation_db(self, path):
        rotation_dict = {}
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            filepath = str(r["path"])
            if r["score"] > 0.9 and r["rot"] > 0:
                rotation_dict[filepath] = r["rot"]
        return rotation_dict

    def __getitem__(self, index):
        path = self.paths[index]
        assert Path(path).exists()
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        rot = self.rotation_dict.get(str(path), 0)
        if rot > 0:
            if rot == 1:
                im = np.rot90(im, 3)
            elif rot == 2:
                im = np.rot90(im, 2)
            elif rot == 3:
                im = np.rot90(im, 1)

        assert im is not None
        im = im[..., ::-1]

        if self.resize_shape is not None:
            im = cv2.resize(im, self.resize_shape)

        if self.transform:
            im = self.transform(image=im)["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        dummy_target = torch.tensor(0).long()
        return im, dummy_target

    def __len__(self):
        return len(self.paths)


class HotelTestDataset(Dataset):
    def __init__(self,
                 filepath_list,
                 transform=None,
                 size_template=None,
                 resize_shape=(320, 240)):
        self.paths = filepath_list
        self.transform = transform
        self.resize_shape = resize_shape

    def __getitem__(self, index):
        path = self.paths[index]
        assert Path(path).exists()
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        assert im is not None
        im = im[..., ::-1]

        if self.resize_shape is not None:
            im = cv2.resize(im, self.resize_shape)

        if self.transform:
            im = self.transform(image=im)["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        dummy_target = torch.tensor(0).long()
        return im, dummy_target

    def __len__(self):
        return len(self.paths)


class HotelRotationFixedDataset(Dataset):
    def __init__(self,
                 dataframe_filepath,
                 rotation_fixed_filepath,
                 transform=None,
                 size_template=None,
                 resize_shape=(320, 240),
                 img_dir="data/input/train_images"):
        self.paths = []
        self.class_ids = []
        self.img_dir = img_dir
        self.__load_path_label(dataframe_filepath)
        self.transform = transform
        self.resize_shape = resize_shape
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

        df = pd.read_csv(path)
        for _, r in df.iterrows():
            image, chain = r["image"], r["chain"]
            # train.csv vs valtrain.csv
            hotel_idx = r["hotel_idx"] if "hotel_idx" in r else r["hotel_id"]
            self.paths.append(f"{img_dir}/{chain}/{image}")
            self.class_ids.append(hotel_idx)

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

        if self.resize_shape is not None:
            im = cv2.resize(im, self.resize_shape)

        if self.transform:
            im = self.transform(image=im)["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        target = torch.tensor(self.class_ids[index]).long()
        return im, target

    def __len__(self):
        return len(self.paths)


class HotelRotationFixedVNDataset(Dataset):
    def __init__(self,
                 dataframe_filepath,
                 transform=None,
                 resize_shape=(320, 240),
                 img_dir="data/input/train_images"):
        self.paths = []
        self.class_ids = []
        self.rots = []

        self.img_dir = img_dir
        self.__load_path_label(dataframe_filepath)
        self.transform = transform
        self.resize_shape = resize_shape

    def __load_path_label(self, path):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            image = r["image"]
            # train.csv vs valtrain.csv
            hotel_idx = r["hotel_idx"] if "hotel_idx" in r else r["hotel_id"]
            self.paths.append(image)
            self.class_ids.append(hotel_idx)
            if r["rot"] > 0 and r["rot_score"] > 0.9:
                self.rots.append(r["rot"])
            else:
                self.rots.append(0)

    def __getitem__(self, index):
        path = self.paths[index]
        assert Path(path).exists()
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        rot = self.rots[index]
        if rot > 0:
            if rot == 1:
                im = np.rot90(im, 3)
            elif rot == 2:
                im = np.rot90(im, 2)
            elif rot == 3:
                im = np.rot90(im, 1)

        assert im is not None
        im = im[..., ::-1]

        if self.resize_shape is not None:
            im = cv2.resize(im, self.resize_shape)

        if self.transform:
            im = self.transform(image=im)["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        target = torch.tensor(self.class_ids[index]).long()
        return im, target

    def __len__(self):
        return len(self.paths)


class HotelDataset(Dataset):
    def __init__(self,
                 dataframe_filepath,
                 transform=None,
                 size_template=None,
                 resize_shape=(320, 240),
                 img_dir="data/input/train_images"):
        self.paths = []
        self.class_ids = []
        self.img_dir = img_dir
        self.__load_path_label(dataframe_filepath)
        self.transform = transform
        self.resize_shape = resize_shape

    def __load_path_label(self, path):
        img_dir = self.img_dir

        df = pd.read_csv(path)
        for _, r in df.iterrows():
            image, chain = r["image"], r["chain"]
            # train.csv vs valtrain.csv
            hotel_idx = r["hotel_idx"] if "hotel_idx" in r else r["hotel_id"]
            self.paths.append(f"{img_dir}/{chain}/{image}")
            self.class_ids.append(hotel_idx)

    def __getitem__(self, index):
        path = self.paths[index]
        assert Path(path).exists()
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        assert im is not None
        im = im[..., ::-1]

        if self.resize_shape is not None:
            im = cv2.resize(im, self.resize_shape)

        if self.transform:
            im = self.transform(image=im)["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        target = torch.tensor(self.class_ids[index]).long()
        return im, target

    def __len__(self):
        return len(self.paths)


def test_dataloader(
    filepath_list,
    test_batch_size=256,
    num_workers=8,
    test_transform=None,
    resize_shape=(320, 240),
):
    ds_test = HotelTestDataset(filepath_list, transform=test_transform,
                               resize_shape=resize_shape)
    return DataLoader(
        dataset=ds_test,
        sampler=SequentialSampler(ds_test),
        batch_size=test_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )


def train_dataloaders(
    valtrain_path: str,
    valtest_path: str,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    num_workers: int = 8,
    train_transform=None,
    val_transform=None,
    resize_shape=(320, 240),
    is_distributed=False,
    img_dir="data/input/train_images",
):
    ds_trn = HotelDataset(valtrain_path, transform=train_transform,
                          resize_shape=resize_shape,
                          img_dir=img_dir)
    ds_val = HotelDataset(valtest_path, transform=val_transform,
                          resize_shape=resize_shape,
                          img_dir=img_dir)
    dataloaders = {}
    dataloaders["train"] = DataLoader(
        dataset=ds_trn,
        sampler=RandomSampler(ds_trn) if not is_distributed else (
            DistributedSampler(ds_trn)
        ),
        batch_size=train_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    dataloaders["val"] = DataLoader(
        dataset=ds_val,
        sampler=SequentialSampler(ds_val),
        batch_size=val_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )

    return dataloaders


def index_seq_dataloader(
    valtrain_path: str,
    index_batch_size: int = 32,
    num_workers: int = 8,
    index_transform=None,
    resize_shape=(512, 512),
    img_dir="data/input/train_images",
):
    ds_trn = HotelRotationFixedDataset(
        valtrain_path,
        "data/working/v80_rotation_hotelid_train_images.csv",
        transform=index_transform,
        resize_shape=resize_shape,
        img_dir=img_dir
    )
    return DataLoader(
        dataset=ds_trn,
        sampler=SequentialSampler(ds_trn),
        batch_size=index_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


def index_dataloader(
    valtrain_path: str,
    index_batch_size: int = 32,
    num_workers: int = 8,
    index_transform=None,
    resize_shape=(320, 240),
    is_distributed=False,
    img_dir="data/input/train_images",
):
    ds_trn = HotelDataset(valtrain_path, transform=index_transform,
                          resize_shape=resize_shape,
                          img_dir=img_dir)
    return DataLoader(
        dataset=ds_trn,
        sampler=RandomSampler(ds_trn) if not is_distributed else (
            DistributedSampler(ds_trn)
        ),
        batch_size=index_batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


def get_dataloaders(train_transform, test_transform, params):
    dataloaders = train_dataloaders(
        "data/input/valtrain.csv",
        "data/input/valtest.csv",
        train_batch_size=params["train_batch_size"],
        val_batch_size=params["val_batch_size"],
        num_workers=params["num_workers"],
        train_transform=train_transform,
        val_transform=test_transform,
        resize_shape=None,
        is_distributed=(params.get("api_version", 1) == 2) and (params.get("num_gpus", 1) > 1),
    )
    dl_index = index_dataloader(
        "data/input/valtrain.csv",
        index_batch_size=params["val_batch_size"],
        num_workers=params["num_workers"],
        index_transform=test_transform,
        resize_shape=None,
        is_distributed=(params.get("api_version", 1) == 2) and (params.get("num_gpus", 1) > 1),
    )
    dataloaders["index"] = dl_index

    return dataloaders
