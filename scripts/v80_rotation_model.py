from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from peko.augmentation import get_augv5_no_flip
from peko.utils.logger import set_logger
from peko.utils.saver import save_checkpoint
from peko.utils.timer import AverageMeter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

logger = getLogger("peko")

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_train_test_split():
    TRAIN_DIR = "data/input/external/hotels50k_rev2/train/"
    orig_train_paths = list(sorted(Path(TRAIN_DIR).glob("*/*/travel_website/*.jpg")))

    train_paths, test_paths = train_test_split(orig_train_paths, train_size=0.8, random_state=1, shuffle=True)
    return train_paths, test_paths


class Hotels50kRotationDataset(Dataset):
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

        # Random rotations
        rot_k = np.random.choice([0, 1, 2, 3])
        if rot_k > 0:
            im = np.rot90(im, rot_k)

        target = torch.tensor(rot_k).long()

        if self.transform is not None:
            im = self.transform(image=im)["image"]
        im = torch.from_numpy(im.transpose((2, 0, 1))).float()

        return im, target


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


def test_main():
    backbone_name = "resnet50"

    weight_path = "data/working/models/v80_rotation_model/"
    weight_path += "v80_rotation_model_epoch08.pth"
    model = timm.create_model(backbone_name, pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()
    model.eval()

    eval_paths = list(sorted([
        str(p)
        for p in Path("data/input/train_images/").glob("*/*.jpg")
    ]))

    batch_size = 64
    input_size = 256
    _, tst_transform = get_augv5_no_flip(input_size)
    tst_ds = HotelIDInferRotationDataset(eval_paths, transform=tst_transform)
    tst_dl = DataLoader(tst_ds, sampler=SequentialSampler(tst_ds), batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=False)

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

        if idx % 100 == 0:
            print(idx, len(tst_dl))

    pd.DataFrame(ret)[[
        "path",
        "rot",
        "score",
    ]].to_csv("data/working/v80_rotation_hotelid_train_images.csv", index=False)


def test_hotelidv3():
    backbone_name = "resnet50"

    weight_path = "data/working/models/v80_rotation_model/"
    weight_path += "v80_rotation_model_epoch08.pth"
    model = timm.create_model(backbone_name, pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(weight_path)["state_dict"])
    model = model.cuda()
    model.eval()

    eval_paths = []
    df = pd.read_csv("data/working/feats/v110_hidv2_hotelidv2.csv")
    for _, r in df.iterrows():
        source = r["source"]
        if source == "hotel-id":
            path = r["image"]
            chain = int(r["chain"])
            eval_paths.append(f"data/input/train_images/{chain}/{path}")
        else:
            path = r["image"]
            if Path(path).exists():
                eval_paths.append(path)
    print(len(eval_paths))

    batch_size = 64
    input_size = 256
    _, tst_transform = get_augv5_no_flip(input_size)
    tst_ds = HotelIDInferRotationDataset(eval_paths, transform=tst_transform)
    tst_dl = DataLoader(tst_ds, sampler=SequentialSampler(tst_ds), batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=False)

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

        if idx % 100 == 0:
            print(idx, len(tst_dl))

    pd.DataFrame(ret)[[
        "path",
        "rot",
        "score",
    ]].to_csv("data/working/v80_rotation_hotelidv3_train_images.csv", index=False)


def main():
    set_logger(logger)

    train_paths, test_paths = get_train_test_split()
    input_size = 224
    init_lr = 1e-4
    total_epochs = 10
    batch_size = 64 * 3

    trn_transform, tst_transform = get_augv5_no_flip(input_size)
    trn_ds = Hotels50kRotationDataset(train_paths, transform=trn_transform)
    trn_dl = DataLoader(trn_ds, sampler=RandomSampler(trn_ds), batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=True)

    backbone_name = "resnet50"
    model = timm.create_model(backbone_name, pretrained=True, num_classes=4)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(trn_dl) * total_epochs, eta_min=1e-6)

    report_interval = 100
    model.train()

    for epoch in range(total_epochs):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for idx, (X, y) in enumerate(trn_dl):
            X, y = X.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), X.size(0))
            acc_meter.update(accuracy(y, out), X.size(0))
            if idx % report_interval == 0:
                logger.info("ep={} ({}/{}) loss={:.6f}, acc={:.6f}".format(
                    epoch,
                    idx,
                    len(trn_dl),
                    loss_meter.avg,
                    acc_meter.avg,
                ))

        weight_path = "data/working/models/v80_rotation_model/"
        Path(weight_path).mkdir(parents=True, exist_ok=True)
        weight_path += f"v80_rotation_model_epoch{epoch + 1:02d}.pth"
        save_checkpoint(
            path=str(weight_path),
            model=model,
            epoch=epoch,
            optimizer=optimizer,
        )


if __name__ == "__main__":
    main()
    test_main()
    test_hotelidv3()
