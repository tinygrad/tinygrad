import torch
import torchvision
from tqdm import tqdm
import time
import wandb
from extra.datasets.imagenet import iterate, get_train_files, get_val_files


seed = 42
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torchvision.models.resnet50().to(device)

BS = 16
lr = 0.256 * (BS / 256)  # Linearly scale from BS=256, lr=0.256
epochs = 100
lf = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

def train_step(X, Y):
    optimizer.zero_grad()
    out = model.forward(X)
    loss = lf(out, Y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss, out

wandb.init()
steps_in_train_epoch = (len(get_val_files()) // BS) - 1
for e in range(epochs):
    # train loop
    model.train()
    for X, Y, data_time in (t := tqdm(iterate(bs=BS, val=True, num_workers=16), total=steps_in_train_epoch)):
        st = time.monotonic()
        X, Y = torch.tensor(X).to(device), torch.tensor(Y).to(device)
        loss, out = train_step(X, Y)
        et = time.monotonic()
        loss_cpu = loss.cpu().detach().numpy()
        cl = time.monotonic()

        print(f"{(data_time+cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms CL, {data_time*1000.0:7.2f} ms fetch data, {loss_cpu:7.2f} loss, {torch.cuda.max_memory_allocated()/1e9:.2f} GB used")
        wandb.log({"lr": scheduler.get_last_lr(),
                    "train/data_time": data_time,
                    "train/python_time": et - st,
                    "train/step_time": cl - st,
                    "train/other_time": cl - et,
                    "train/loss": loss_cpu,
        })
