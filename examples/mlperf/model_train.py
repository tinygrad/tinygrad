from pathlib import Path
from tqdm import tqdm
from tinygrad.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.nn import optim

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
  from datasets.wikipedia import iterate
  from transformers import BertTokenizer

  mdl = BertForPreTraining()
  mdl.load_from_pretrained()

  tokenizer = BertTokenizer(str(Path(__file__).parent.parent.parent / "weights/bert_vocab.txt"))

  params = optim.get_parameters(mdl)
  optimizer = optim.SGD(params, lr=0.001)

  @TinyJit
  def train_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels):
    pred, seq = mdl(input_ids, input_mask, segment_ids)
    loss = mdl.loss(pred, seq, masked_lm_positions, masked_lm_ids, next_sentence_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.realize()

  for i in range(1000):
    for X, Y in (t := tqdm(iterate(tokenizer), total=53)):
      input_ids, input_mask, segment_ids = Tensor(X["input_ids"], requires_grad=False), Tensor(X["input_mask"], requires_grad=False), Tensor(X["segment_ids"], requires_grad=False)
      masked_lm_positions, masked_lm_ids, next_sentence_labels = Tensor(X["masked_lm_positions"], requires_grad=False), Tensor(X["masked_lm_ids"], requires_grad=False), Tensor(X["next_sentence_labels"], requires_grad=False)
      loss = train_step(input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels + 1 - 1)
      t.set_description(f"loss {loss.numpy().item():.2f}")

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


