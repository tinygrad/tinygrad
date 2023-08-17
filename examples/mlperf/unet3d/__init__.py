from dataclasses import dataclass
from models.unet3d import UNet3D
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
    

def main(flags):
    model = UNet3D(1, 3, normalization=flags.normalization, activation=flags.activation)
    
    train_dataloader, val_dataloader = get_data_loaders(flags) # todo: multi-gpu
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