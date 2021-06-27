from .hotelid import get_dataloaders as get_dataloaders_hotelid
from .hotelid_rf_v2 import get_dataloaders as get_dataloaders_hotelid_rf_v2
from .hotelid_rf_v3 import get_dataloaders as get_dataloaders_hotelid_rf_v3
from .hotels50k_rev2 import get_dataloaders as get_dataloaders_hotels50k_rev2
from .hotels50k_s1024 import get_dataloaders as get_dataloaders_hotels50k_s1024

loaders = {
    "hotelid": get_dataloaders_hotelid,
    "hotels50k_s1024": get_dataloaders_hotels50k_s1024,
    "hotels50k_rev2": get_dataloaders_hotels50k_rev2,
    "hotelid_rf_v2": get_dataloaders_hotelid_rf_v2,
    "hotelid_rf_v3": get_dataloaders_hotelid_rf_v3,
}


def get_dataloaders(dataset_name, train_transform, val_transform, params):
    if dataset_name in loaders:
        return loaders[dataset_name](train_transform, val_transform, params)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
