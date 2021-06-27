import sys
from logging import INFO, FileHandler, Formatter, StreamHandler


def set_logger(logger):
    logformat = "%(asctime)s %(levelname)s %(message)s"
    handler_out = StreamHandler(sys.stdout)
    handler_out.setLevel(INFO)
    handler_out.setFormatter(Formatter(logformat))
    logger.setLevel(INFO)
    logger.addHandler(handler_out)


def add_log_filehandler(logger, conf_name: str, logfile_path: str):
    logformat = "%(asctime)s %(levelname)s %(message)s"
    handler = FileHandler(logfile_path)
    handler.setLevel(INFO)
    handler.setFormatter(Formatter(logformat))
    logger.addHandler(handler)
