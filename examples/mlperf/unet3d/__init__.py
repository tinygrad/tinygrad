from dataclasses import dataclass
from typing import Optional
from examples.mlperf.unet3d.data_loader import get_data_loaders
from examples.mlperf.unet3d.inference import evaluate
from examples.mlperf.unet3d.training import train
from models.unet3d import Unet3D
from .losses import DiceCELoss, DiceScore
from math import ceil

DATASET_SIZE = 168

@dataclass
class Flags:
    data_dir: str = "./extra/dataset/kits19/data"
    input_shape: tuple[int, int, int] = (128, 128, 128)
    oversampling: float = 0.4
    seed: int = 42
    shuffling_seed: int = 42
    batch_size: int = 2
    benchmark: bool =False
    num_workers: int = 1
    max_epochs: int = 4000
    quality_threshold: float = 0.908
    start_eval_at: int = 1000
    evaluate_every: int = 20
    learning_rate: float = 0.8
    init_learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_warmup_epochs: int = 200
    lr_decay_epochs: Optional[int] = None
    gradient_accumulation_steps: int = 1
    include_background: bool = False
    overlap: float = 0.5 # inference
    val_input_shape: tuple[int, int, int] = (128, 128, 128)
    verbose: bool = True
    

def main(flags):
    model = Unet3D(1, 3)
    
    train_dataloader, val_dataloader = get_data_loaders(flags, 1, 0) # todo: multi-gpu
    samples_per_epoch = len(train_dataloader) * flags.batch_size
    
    flags.evaluate_every = flags.evaluate_every or ceil(20*DATASET_SIZE/samples_per_epoch)
    flags.start_eval_at = flags.start_eval_at or ceil(1000*DATASET_SIZE/samples_per_epoch)
    
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout,
                         include_background=flags.include_background)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout=flags.layout,
                         include_background=flags.include_background)
    
    if flags.exec_mode == 'train':
        train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn)

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model, val_dataloader, loss_fn, score_fn)
        for key in eval_metrics.keys():
                print(key, eval_metrics[key])