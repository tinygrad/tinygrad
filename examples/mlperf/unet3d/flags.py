from dataclasses import dataclass
from typing import Optional, Tuple

DATASET_SIZE = 168

@dataclass
class Flags:
    data_dir: str = "./extra/dataset/kits19/data"
    input_shape: tuple[int, int, int] = (128, 128, 128)
    val_input_shape: tuple[int, int, int] = (128, 128, 128)
    oversampling: float = 0.4
    seed: int = 42
    shuffling_seed: int = 42
    batch_size: int = 2
    benchmark: bool = False
    num_workers: int = 1
    max_epochs: int = 4000
    quality_threshold: float = 0.908
    start_eval_at: int = 200
    evaluate_every: int = 20
    learning_rate: float = 0.8
    init_learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_warmup_epochs: int = 200
    lr_decay_epochs: Optional[Tuple[int]] = None
    lr_decay_factor: float = 1.0
    gradient_accumulation_steps: int = 1
    include_background: bool = False
    overlap: float = 0.5 # inference
    verbose: bool = True