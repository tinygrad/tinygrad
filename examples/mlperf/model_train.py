from extra import dist
if __name__ == "__main__":
  dist.preinit()

from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tinygrad.state import safe_save, get_state_dict
from extra.dist import collectives
import wandb

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  from models.bert import BertForPreTraining
  from extra.datasets.wikipedia import iterate, get_train_files
  from extra.lr_scheduler import CosineAnnealingLR
  from extra.dist import OOB

  rank, world_size = getenv("RANK"), getenv("WORLD_SIZE")

  mdl = BertForPreTraining()
  mdl.load_from_pretrained()

  params = get_state_dict(mdl)
  optimizer = optim.LAMB(list(params.values()), lr=1e-4, wd=0.01)
  lr_scheduler = CosineAnnealingLR(optimizer, len(get_train_files()), eta_min=1e-6)

  if rank == 0:
    wandb.init(project="tinygrad-mlperf")

  @TinyJit
  def eval_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids):
    pred, _ = mdl(input_ids, input_mask, segment_ids)
    acc = mdl.accuracy(pred, masked_lm_positions, masked_lm_ids)
    return acc.realize()

  @TinyJit
  def train_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels):
    pred, seq = mdl(input_ids, input_mask, segment_ids)
    loss = mdl.loss(pred, seq, masked_lm_positions, masked_lm_ids, next_sentence_labels)
    optimizer.zero_grad()
    loss.backward()

    # sync gradients across ranks
    bucket, bucket_meta, offset = [], {}, 0
    for k, v in params.items():
      if v.grad is not None:
        bucket_meta[k] = (v.numel(), v.shape)
        bucket.append(v.grad.flatten())
    grads = collectives.allreduce(Tensor.cat(*bucket), cache_id="grads")
    for k in bucket_meta:
      size = bucket_meta[k][0]
      params[k].grad.assign(grads[offset:offset+size].reshape(*bucket_meta[k][1]))
      offset += size

    optimizer.step()
    return loss.realize()

  print(f"there are {len(get_train_files())} training files")

  done = False
  for i in range(1000):
    # train loop
    for j, (X, _) in enumerate(iterate(bs=8)):
      input_ids, input_mask, segment_ids = Tensor(X["input_ids"], requires_grad=False), Tensor(X["input_mask"], requires_grad=False), Tensor(X["segment_ids"], requires_grad=False)
      masked_lm_positions, masked_lm_ids, next_sentence_labels = Tensor(X["masked_lm_positions"], requires_grad=False), Tensor(X["masked_lm_ids"], requires_grad=False), Tensor(X["next_sentence_labels"], requires_grad=False)
      # split across ranks
      input_ids, input_mask, segment_ids = input_ids.chunk(world_size, 0)[rank], input_mask.chunk(world_size, 0)[rank], segment_ids.chunk(world_size, 0)[rank]
      masked_lm_positions, masked_lm_ids, next_sentence_labels = masked_lm_positions.chunk(world_size, 0)[rank], masked_lm_ids.chunk(world_size, 0)[rank], next_sentence_labels.chunk(world_size, 0)[rank]
      loss = train_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels)
      lr_scheduler.step()

      # update wandb
      if rank == 0:
        wandb.log({
          "loss": loss.numpy().item(),
          "lr": optimizer.lr.numpy().item(),
        })

      # save checkpoint every 10000 steps
      if j % 10000 == 0 and rank == 0:
        safe_save(get_state_dict(mdl), f"weights/ckpt_{i}_{j}.bert.safetensors")

      # eval loop
      if j % 150000 == 0:
        Tensor.training = False
        train_step.jit_cache = []
        train_step.cnt = 0
        accuracies = []
        for X, _ in iterate(bs=8, val=True):
          input_ids, input_mask, segment_ids = Tensor(X["input_ids"], requires_grad=False), Tensor(X["input_mask"], requires_grad=False), Tensor(X["segment_ids"], requires_grad=False)
          masked_lm_positions, masked_lm_ids = Tensor(X["masked_lm_positions"], requires_grad=False), Tensor(X["masked_lm_ids"], requires_grad=False)
          input_ids, input_mask, segment_ids = input_ids.chunk(world_size, 0)[rank], input_mask.chunk(world_size, 0)[rank], segment_ids.chunk(world_size, 0)[rank]
          masked_lm_positions, masked_lm_ids = masked_lm_positions.chunk(world_size, 0)[rank], masked_lm_ids.chunk(world_size, 0)[rank]
          acc = eval_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids).numpy().item()
          accuracies.append(acc)

        acc_sum, acc_len = sum(accuracies), len(accuracies)
        if rank == 0:
          for l in range(1, world_size):
            recv_sum, recv_len = OOB.recv(l)
            acc_sum += recv_sum
            acc_len += recv_len
        elif rank < world_size:
          OOB.send((acc_sum, acc_len), 0)

        if rank == 0:
          final_acc = acc_sum / acc_len
          wandb.log({"eval_accuracy": final_acc})
          if final_acc >= 0.72:
            print("done training")
            done = True
            # broadcast done signal
            for l in range(1, world_size):
              OOB.send(True, l)
            break
          else:
            # broadcast done signal
            for l in range(1, world_size):
              OOB.send(False, l)
        else:
          done = OOB.recv(0)
        eval_step.jit_cache = []
        eval_step.cnt = 0
        Tensor.training = True
    if done: break

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

def train():
  Tensor.training = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
    nm = f"train_{m}"
    if nm in globals():
      print(f"training {m}")
      globals()[nm]()

if __name__ == "__main__":
  devices = ["gpu:0", "gpu:1", "gpu:2", "gpu:3"]
  world_size = len(devices)

  # init out-of-band communication
  dist.init_oob(world_size)

  # start the processes
  processes = []
  for rank, device in enumerate(devices):
    processes.append(dist.spawn(rank, device, fn=train, args=()))
  for p in processes: p.join()
