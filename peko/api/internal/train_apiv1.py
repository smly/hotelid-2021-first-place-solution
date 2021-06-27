import argparse
from logging import getLogger
from pathlib import Path

import peko.torch.metrics
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from peko.cbir.search import (compute_mapk_score, extract_features,
                              search_with_faiss_cpu)
from peko.dataset.factory import get_dataloaders
from peko.path import get_last_model_path, get_logpath
from peko.torch.angular import AngularModel
from peko.utils.colab import is_colab
from peko.utils.common import dynamic_load, load_optimizer, load_scheduler
from peko.utils.configs import load_config
from peko.utils.feature_container import GlobalFeatureContainer
from peko.utils.logger import add_log_filehandler, set_logger
from peko.utils.saver import save_checkpoint
from peko.utils.timer import AverageMeter, timer
from tensorboardX import SummaryWriter
from torch.cuda import amp

logger = getLogger("peko")


def load_model(conf):
    with timer("Init model...", logger=logger):
        # TODO(smly): もしモデルの定義がコンフィグにあれば使う
        if "model" in conf:
            model_kwargs = conf.model.kwargs
            model_cls = dynamic_load(conf.model.fqdn)
            model = model_cls(**model_kwargs)
        else:
            model = AngularModel(n_classes=conf["n_classes"],
                                 model_name=conf["backbone"],
                                 pretrained=True)

        # Fine-tuning
        if conf["pretrained_model"]:
            model = load_pretrained_model(model, conf)

        model = model.cuda()

        # Local=DataParallel, CoLab=Plane.
        if not is_colab():
            model = nn.DataParallel(model)
    return model


def load_pretrained_model(model, conf):
    model_weights = torch.load(conf["pretrained_model"])
    if conf.get("pretrained_remove_fc", True):
        ignore_keys = [
            "fc.weight", "fc.bias", "bn.weight", "bn.bias",
            "bn.running_mean", "bn.running_var",
            "bn.num_batches_tracked", "final.weight",
        ]
        for k in list(model_weights["state_dict"].keys()):
            if k in ignore_keys:
                del model_weights["state_dict"][k]
        model.load_state_dict(
            model_weights["state_dict"], strict=False)
    else:
        model.load_state_dict(
            model_weights["state_dict"], strict=True)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--auto-resume", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    conf = load_config(args.config)
    conf_name = conf.name
    use_amp = conf.get("use_amp", False)

    # Initialize.
    logfile_path, scalars_json_path = get_logpath(conf_name)
    set_logger(logger)
    add_log_filehandler(logger, conf_name, logfile_path)
    logger.info(conf_name + " :=\n" + OmegaConf.to_yaml(conf))

    get_aug = dynamic_load(conf["augmentation_func"])

    with timer("Build dataloaders ... ", logger=logger):
        train_transform, val_transform = get_aug(conf["input_size"])
        dataloaders = get_dataloaders(conf.train_dataset, train_transform, val_transform, conf)

    evaluation_set = {}
    with timer("Build dataloaders for evaluation ... ", logger=logger):
        for eval_dataset_name in conf.eval_dataset:
            logger.info(f" - Processing for {eval_dataset_name}")
            train_transform, val_transform = get_aug(conf["input_size"])
            eval_dataloaders = get_dataloaders(
                eval_dataset_name, train_transform, val_transform, conf)
            evaluation_set[eval_dataset_name] = eval_dataloaders

    model = load_model(conf)
    total_epochs = conf["total_epochs"]
    total_iters = total_epochs * len(dataloaders["train"])

    optimizer = load_optimizer(model.parameters(), conf)
    scheduler = load_scheduler(optimizer, conf)
    if conf["scheduler"]["fqdn"].endswith("CosineAnnealingLR"):
        scheduler.T_max = total_iters

    criterion = nn.CrossEntropyLoss()

    num_images = len(dataloaders["train"].dataset)
    num_iters_per_epoch = len(dataloaders["train"])
    logger.info(f"Num images: {num_images}, iters per epoch: {num_iters_per_epoch}")

    writer = SummaryWriter(f"data/working/tb_logs/{conf_name}/")
    scaler = amp.GradScaler()
    n_accumulate = 1

    for epoch in range(total_epochs):
        trn_meters = {
            "loss": AverageMeter(),
            "prec": AverageMeter(),
        }
        model.train(True)

        for idx, (X, y) in enumerate(dataloaders["train"]):
            X, y = X.to("cuda"), y.to("cuda")
            batch_size = X.size(0)

            optimizer.zero_grad()

            if use_amp:
                with amp.autocast(enabled=True):
                    outputs = model(X, y)
                    loss = criterion(outputs, y)
                    loss = loss / n_accumulate
                    scaler.scale(loss).backward()

                loss.backward()

                if (idx+1) % n_accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()

            else:
                outputs = model(X, y)
                loss = criterion(outputs, y)
                loss.backward()

                optimizer.step()
                scheduler.step()

            acc = peko.torch.metrics.accuracy(outputs, y)
            trn_meters["loss"].update(loss.item(), batch_size)
            trn_meters["prec"].update(acc, batch_size)

            if idx % 20 == 0:
                logger.info(
                    "train "
                    + "{:.4f}".format(epoch + idx/num_iters_per_epoch) + " "
                    + "{:.8f}".format(trn_meters["loss"].avg) + " "
                    + "{:.8f}".format(trn_meters["prec"].avg))
                writer.add_scalar(
                    "train/loss", trn_meters["loss"].avg,
                    epoch * num_iters_per_epoch + idx)
                writer.add_scalar(
                    "train/acc", trn_meters["prec"].avg,
                    epoch * num_iters_per_epoch + idx)
                writer.add_scalar(
                    "lr", optimizer.param_groups[0]["lr"], epoch * num_iters_per_epoch + idx)

        logger.info(
            "train "
            + "{:.4f}".format(epoch + 1) + " "
            + "{:.8f}".format(trn_meters["loss"].avg) + " "
            + "{:.8f}".format(trn_meters["prec"].avg))

        if "val" in dataloaders:
            val_meters = {
                "loss": AverageMeter(),
                "prec": AverageMeter(),
            }
            model.eval()
            for idx, (X, y) in enumerate(dataloaders["val"]):
                X, y = X.to("cuda"), y.to("cuda")
                batch_size = X.size(0)

                with torch.no_grad():
                    outputs = model(X, y)
                    loss = criterion(outputs, y)
                    acc = peko.torch.metrics.accuracy(outputs, y)

                val_meters["loss"].update(loss.item(), batch_size)
                val_meters["prec"].update(acc, batch_size)

            writer.add_scalar("val/loss", val_meters["loss"].avg, (epoch + 1) * num_iters_per_epoch)
            writer.add_scalar("val/acc", val_meters["prec"].avg, (epoch + 1) * num_iters_per_epoch)

            logger.info(
                "val "
                + "{:.4f}".format(epoch + 1) + " "
                + "{:.8f}".format(val_meters["loss"].avg) + " "
                + "{:.8f}".format(val_meters["prec"].avg))

        weight_path = get_last_model_path(conf, epoch=epoch + 1)
        save_checkpoint(
            path=str(weight_path),
            model=model,
            epoch=epoch,
            optimizer=optimizer,
        )

        # Extract features
        logger.info("Extract features ...")
        model_single = model.module
        for eval_dataset_name in conf.eval_dataset:
            feature_dirname = Path(weight_path).stem
            feature_path = Path("data/working/feats") / feature_dirname / f"{eval_dataset_name}.h5"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            if not feature_path.exists():
                gfc = GlobalFeatureContainer(feature_path)
                # Extract features from dataloaders["index"] and dataloaders["val"]
                extract_features(
                    model_single,
                    evaluation_set[eval_dataset_name],
                    gfc,
                    target_list=["index", "val"])

        # Search with extracted features
        logger.info("Search with extracted features ...")
        for eval_dataset_name in conf.eval_dataset:
            feature_dirname = Path(weight_path).stem
            feature_path = Path("data/working/feats") / feature_dirname / f"{eval_dataset_name}.h5"
            gfc = GlobalFeatureContainer(feature_path)

            feat_index, feat_test = gfc.get("feat_index"), gfc.get("feat_val")
            label_index, label_test = gfc.get("label_index"), gfc.get("label_val")
            dists, topk_idx = search_with_faiss_cpu(feat_test, feat_index, topk=5)
            mapk_score = compute_mapk_score(label_test, label_index, dists, topk_idx, topk=5)
            logger.info(f"{feature_dirname}/{eval_dataset_name} MAP@5: {mapk_score:.6f}")
            writer.add_scalar(f"eval/{eval_dataset_name}", mapk_score, epoch + 1)
