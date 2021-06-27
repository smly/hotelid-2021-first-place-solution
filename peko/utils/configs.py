from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config: str) -> DictConfig:
    conf = OmegaConf.load(config)
    conf.name = Path(config).stem

    OmegaConf.set_readonly(conf, False)
    return conf
