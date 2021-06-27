import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from logging import getLogger

import cv2
from peko.api.internal.train_apiv1 import main as main_v1
from peko.utils.configs import load_config

logger = getLogger("peko")

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--auto-resume", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    conf = load_config(args.config)
    api_version = conf.get("api_version", 1)

    if api_version == 1:
        main_v1(args)
    else:
        raise RuntimeError


if __name__ == "__main__":
    main(parse_args())
