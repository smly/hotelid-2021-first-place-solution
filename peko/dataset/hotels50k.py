import pickle
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def get_sampled_train_dataframe(df_train, max_freq=100, min_class_freq=10):
    valcnt = df_train["hotel_id"].value_counts()
    df_train = df_train[df_train["hotel_id"].isin(
        valcnt[valcnt >= min_class_freq].index.tolist())]

    df_part_list = []
    for hotel_id, df_part in df_train.groupby("hotel_id"):
        df_fixed = df_part[df_part["im_source"] == "traffickcam"]
        if len(df_fixed) > 0:
            df_part_list.append(df_fixed)

        df_remaining = df_part[df_part["im_source"] != "traffickcam"]
        n_remaining = len(df_remaining)
        if n_remaining > 0:
            max_freq_ = max_freq - len(df_fixed)
            df_part_list.append(df_remaining.sample(min(n_remaining, max_freq_)))

    return pd.concat(df_part_list)


def get_class_mapping(hotel_id_unique):
    mapping = {}
    for idx, hotel_id in enumerate(sorted(hotel_id_unique)):
        mapping[hotel_id] = idx
    return mapping


class Hotel50kTrainDataset(Dataset):
    def __init__(self, df_train, mapping, transforms):
        images, labels = self.covnert_to_path_list(df_train)
        self.transforms = transforms
        self.mapping = mapping
        self.images = images
        self.labels = [mapping[lbl] for lbl in labels]
        assert len(self.mapping.keys()) == 38768

    def covnert_to_path_list(self, df_train):
        return [
            "data/input/external/Hotels-50K/images/{}/{}/{}/{}/{}.jpg".format(
                "train" if r["is_train"] else "test",
                r["chain_id"],
                r["hotel_id"],
                r["im_source"],
                r["im_id"],
            )
            for _, r in df_train.iterrows()
        ], [
            r["hotel_id"]
            for _, r in df_train.iterrows()
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]

        if not Path(path).exists():
            print("path", path)

        assert Path(path).exists()
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        assert im is not None

        im = self.transforms(image=im)["image"]
        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        target = torch.tensor(label).long()
        return im, target


def get_hotel50k_dataset():
    with open("data/input/external/Hotels-50K/hotels50k_orig.pkl", "rb") as f:
        data = pickle.load(f)

    df_all = data["all"]
    val_ids = data["val_close_im_ids"]
    df_val = df_all[df_all["im_id"].isin(val_ids)]

    # n_orig_train_count = len(data["train"])
    df_train = get_sampled_train_dataframe(data["train"])
    # n_sampled_train_count = len(df_train)
    # print("Sampled train images: {} -> {}".format(n_orig_train_count, n_sampled_train_count))

    # _classes = df_train["hotel_id"].nunique()
    hotel_id_classidx_mapping = get_class_mapping(df_train["hotel_id"].unique())
    # print("Num Classes: {}".format(n_classes))

    df_val = df_val[df_val["hotel_id"].isin(hotel_id_classidx_mapping.keys())]
    return df_train, df_val, hotel_id_classidx_mapping


def get_dataloaders(train_transform, test_transform, params):
    df_train, df_val, hotel_id_classidx_mapping = get_hotel50k_dataset()
    df_index = df_train[df_train["im_source"] == "traffickcam"]

    ds_trn = Hotel50kTrainDataset(df_train, hotel_id_classidx_mapping, train_transform)
    ds_val = Hotel50kTrainDataset(df_val, hotel_id_classidx_mapping, test_transform)
    ds_index = Hotel50kTrainDataset(df_index, hotel_id_classidx_mapping, test_transform)

    dataloaders = {}
    dataloaders["train"] = DataLoader(
        dataset=ds_trn,
        sampler=RandomSampler(ds_trn),
        batch_size=params["train_batch_size"],
        pin_memory=True,
        num_workers=params["num_workers"],
        drop_last=True,
    )
    dataloaders["val"] = DataLoader(
        dataset=ds_val,
        sampler=SequentialSampler(ds_val),
        batch_size=params["val_batch_size"],
        pin_memory=True,
        num_workers=params["num_workers"],
        drop_last=False,
    )
    dataloaders["index"] = DataLoader(
        dataset=ds_index,
        sampler=SequentialSampler(ds_index),
        batch_size=params["val_batch_size"],
        pin_memory=True,
        num_workers=params["num_workers"],
        drop_last=False,
    )

    return dataloaders
