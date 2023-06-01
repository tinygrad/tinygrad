from pathlib import Path
from tqdm import tqdm
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
  optimizer = optim.LAMB(params, lr=0.001)
  for i in range(1000):
    for X, Y in (t := tqdm(iterate(tokenizer))):
      pred, seq = mdl(Tensor(X["input_ids"]), Tensor(X["input_mask"]), Tensor(X["segment_ids"]))
      loss = mdl.loss(pred, seq, Tensor(X["masked_lm_positions"]), Tensor(X["masked_lm_ids"]), Tensor(X["next_sentence_labels"]))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
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


