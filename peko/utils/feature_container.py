from pathlib import Path

import h5py
import numpy as np


class GlobalFeatureContainer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.keys = set()

    def contains(self, keyname: str):
        if not Path(self.filepath).exists():
            return False
        with h5py.File(self.filepath, "r") as f:
            return keyname in f.keys()

    def add(self, keyname: str, data):
        if self.contains(keyname):
            return
        if Path(self.filepath).exists():
            if not self.contains(keyname):
                with h5py.File(self.filepath, "a") as f:
                    f.create_dataset(keyname, data=data, compression="gzip")
        else:
            if not self.contains(keyname):
                with h5py.File(self.filepath, "w") as f:
                    f.create_dataset(keyname, data=data, compression="gzip")

    def __getitem__(self, keyname: str):
        ret = self.get(keyname)
        assert ret is not None
        return ret

    def get(self, keyname: str):
        if not Path(self.filepath).exists():
            return
        with h5py.File(self.filepath, "r") as f:
            if keyname in f.keys():
                return np.array(f[keyname])
            else:
                return

    def keys(self):
        with h5py.File(self.filepath, "r") as f:
            return f.keys()
