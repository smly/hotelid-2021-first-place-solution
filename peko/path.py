import datetime
from pathlib import Path


def get_logpath(conf_name: str, suffix="train"):
    # Logging filehandler.
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")
    logfile_path = f"data/working/logs/{conf_name}/{timestamp}_{suffix}.log"
    scalars_json_path = f"data/working/logs/{conf_name}/{timestamp}_{suffix}.json"
    Path(logfile_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scalars_json_path).parent.mkdir(parents=True, exist_ok=True)
    return logfile_path, scalars_json_path


def get_last_model_path(conf, epoch=None):
    epoch = epoch or conf.total_epochs
    model_path = (
        Path("data/working/models") / conf.name
        / f"{conf.name}_epoch{epoch:d}.pth")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    return model_path
