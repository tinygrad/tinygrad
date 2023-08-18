from tqdm import tqdm
from examples.mlperf.unet3d import Flags
from examples.mlperf.unet3d.inference import evaluate

from tinygrad.nn import optim
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor

def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    optimizer.lr = init_lr + (lr - init_lr) * scale
        
def train(flags: Flags, model, train_loader, val_loader, loss_fn, score_fn):
    optimizer = optim.SGD(get_parameters(model), lr=flags.learning_rate, momentum=flags.momentum, weight_decay=flags.weight_decay)
    # scaler = GradScaler() # TODO: add grad scaler
    
    next_eval_at = flags.start_eval_at
    
    if flags.lr_decay_epochs:
        raise NotImplementedError("TODO: lr decay")
    
    Tensor.training = True
    for epoch in range(1, flags.max_epochs + 1):
        cumulative_loss = []
        if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
            lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
            
            loss_value = None
            optimizer.zero_grad()
            for iteration, batch in enumerate(tqdm(train_loader, disable=not flags.verbose)):
                image, label = batch
                image = Tensor(image.numpy())
                label = Tensor(label.numpy())
                
                output = model(image)
                loss_value = loss_fn(output, label)
                
                loss_value.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                # loss_value = reduce_tensor(loss_value, world_size).detach().cpu().numpy() # TODO: reduce tensor for distributed training
                cumulative_loss.append(loss_value)
        
        if flags.lr_decay_epochs:
            pass
            # scheduler.step()
            
        if epoch == next_eval_at:
            next_eval_at += flags.evaluate_every
            del output
            
            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, epoch)
            eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)
            
            model.train()
            
            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                is_successful = True
            elif eval_metrics["mean_dice"] < 1e-6:
                print("MODEL DIVERGED. ABORTING.")
                diverged = True
        
        if is_successful or diverged:
            break
   