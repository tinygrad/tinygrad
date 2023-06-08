from tqdm import tqdm
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tinygrad.state import safe_save, get_state_dict

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
  from datasets.wikipedia import iterate, get_val_files
  import wandb

  mdl = BertForPreTraining()
  mdl.load_from_pretrained()

  params = optim.get_parameters(mdl)
  optimizer = optim.SGD(params, lr=0.001)
  optimizer.zero_grad()

  wandb.init(project="tinygrad-mlperf", settings=wandb.Settings(_disable_stats=True))

  @TinyJit
  def eval_step(input_ids, input_mask, segment_ids):
    pred, _ = mdl(input_ids, input_mask, segment_ids)
    return pred.realize()

  @TinyJit
  def train_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels):
    pred, seq = mdl(input_ids, input_mask, segment_ids)
    loss = mdl.loss(pred, seq, masked_lm_positions, masked_lm_ids, next_sentence_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.realize()

  for i in range(1000):
    # train loop
    eval_step.jit_cache = []
    train_step.jit_cache = []
    Tensor.training = True
    for X, Y in (t := tqdm(iterate(), total=len(get_val_files()))):
      input_ids, input_mask, segment_ids = Tensor(X["input_ids"], requires_grad=False), Tensor(X["input_mask"], requires_grad=False), Tensor(X["segment_ids"], requires_grad=False)
      masked_lm_positions, masked_lm_ids, next_sentence_labels = Tensor(X["masked_lm_positions"], requires_grad=False), Tensor(X["masked_lm_ids"], requires_grad=False), Tensor(X["next_sentence_labels"], requires_grad=False)
      loss = train_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels + 1 - 1)

      # update progress bar
      t.set_description(f"loss {loss.numpy().item():.2f}")

      # update wandb
      wandb.log({
        "loss": loss.numpy().item(),
        "time_remaining": (t.total - t.n) / t.format_dict["rate"] if t.format_dict["rate"] and t.total else 0,
      })

    # eval loop
    eval_step.jit_cache = []
    train_step.jit_cache = []
    Tensor.training = False
    accuracies = []
    for X, Y in (t := tqdm(iterate(val=True), total=len(get_val_files()))):
      input_ids, input_mask, segment_ids = Tensor(X["input_ids"], requires_grad=False), Tensor(X["input_mask"], requires_grad=False), Tensor(X["segment_ids"], requires_grad=False)
      masked_lm_positions, masked_lm_ids = Tensor(X["masked_lm_positions"], requires_grad=False), X["masked_lm_ids"]
      pred = eval_step(input_ids, input_mask, segment_ids)
      acc = mdl.accuracy(pred, masked_lm_positions, masked_lm_ids)
      accuracies.append(acc.item())
      t.set_description(f"acc {acc.item():.8f} avg {sum(accuracies) / len(accuracies):.8f}")

    # save checkpoint
    # safe_save(get_state_dict(mdl), f"weights/ckpt_{i}.bert.safetensors")

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  Tensor.training = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
    nm = f"train_{m}"
    if nm in globals():
      print(f"training {m}")
      globals()[nm]()


