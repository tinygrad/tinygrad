from dataclasses import dataclass
from typing import Optional, Tuple

DATASET_SIZE = 168

@dataclass
class Conf:
    data_dir: str = "./extra/dataset/kits19/data"
    log_dir: str = ""
    save_ckpt_path: str = ""
    load_ckpt_path: str = ""

    epochs: int = 1
    warmup_step: int = 4
    batch_size: int = 2
    optimizer: str = "sgd"
