import tqdm


def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr + (lr - init_lr) * scale
        
def train(flags, model, train_loader, val_loader, loss_fn, score_fn):
    optimizer = ...
    # scaler = GradScaler()
    
    next_eval_at = flags.start_eval_at
    
    if flags.lr_decay_epochs:
        raise NotImplementedError("TODO: lr decay")
    
    model.train()
    for epoch in range(1, flags.epochs + 1):
        cumulative_loss = []
        if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
            lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
            
            loss_value = None
            optimizer.zero_grad()
            for iteration, batch in enumerate(tqdm(train_loader, disable=not flags.verbose)):
                image, label = batch
                
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
            
            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, epoch)
            eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)
            
            model.train()
            
            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                is_successful = True
            elif eval_metrics["mean_dice"] < 1e-6:
                print("MODEL DIVERGED. ABORTING.")
                diverged = True
        
        if is_successful or diverged:
            break