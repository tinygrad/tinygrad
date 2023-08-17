from models.unet3d import UNet3D

DATASET_SIZE = 168

@dataclass
class Flags:
    batch_size: int = 2

def main(flags):
    model = UNet3D(1, 3, normalization=flags.normalization, activation=flags.activation)
    
    train_dataloader, val_dataloader = get_data_loaders(flags, num_shards=world_size, global_rank=local_rank)
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