import time
from pathlib import Path
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.helpers import getenv

def eval_resnet():
  # Resnet50-v1.5
  from tinygrad.jit import TinyJit
  from models.resnet import ResNet50
  mdl = ResNet50()
  mdl.load_from_pretrained()

  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  def input_fixup(x):
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std
    return x

  mdlrun = TinyJit(lambda x: mdl(input_fixup(x)).realize())

  # evaluation on the mlperf classes of the validation set from imagenet
  from datasets.imagenet import iterate
  from extra.helpers import cross_process

  n,d = 0,0
  st = time.perf_counter()
  for x,y in cross_process(iterate):
    dat = Tensor(x.astype(np.float32))
    mt = time.perf_counter()
    outs = mdlrun(dat)
    t = outs.numpy().argmax(axis=1)
    et = time.perf_counter()
    print(f"{(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model")
    print(t)
    print(y)
    n += (t==y).sum()
    d += len(t)
    print(f"****** {n}/{d}  {n*100.0/d:.2f}%")
    st = time.perf_counter()

def eval_retinanet():
  # RetinaNet with ResNeXt50_32X4D
  from models.resnet import ResNeXt50_32X4D
  from models.retinanet import RetinaNet
  mdl = RetinaNet(ResNeXt50_32X4D())
  mdl.load_from_pretrained()

  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  def input_fixup(x):
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std
    return x

  from datasets.openimages import openimages, iterate
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  coco = COCO(openimages())
  coco_results, evaluated_imgs = [], []

  from tinygrad.jit import TinyJit
  mdlrun = TinyJit(lambda x: mdl(input_fixup(x)).realize())

  n, bs = 0, 8
  st = time.perf_counter()
  for x, targets in iterate(coco, bs):
    dat = Tensor(x.astype(np.float32))
    mt = time.perf_counter()
    outs = mdlrun(dat) if dat.shape[0] == bs else mdl(input_fixup(dat)).realize() # jit breaks when batch size changes
    et = time.perf_counter()
    predictions = mdl.postprocess_detections(outs, input_size=dat.shape[1:3], orig_image_sizes=[t["image_size"] for t in targets])
    evaluated_imgs.extend([t["image_id"] for t in targets])
    coco_results.extend([{"image_id": targets[i]["image_id"], "category_id": label, "bbox": box, "score": score}
                    for i, prediction in enumerate(predictions) for box, score, label in zip(*prediction.values())])
    ext = time.perf_counter()
    n += len(targets)
    print(f"[{n}/{len(coco.imgs)}] == {(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model, {(ext-et)*1000:.2f} ms for postprocessing")
    st = time.perf_counter()

  coco_eval = COCOeval(coco, iouType="bbox")
  coco_eval.cocoDt = coco.loadRes(coco_results)
  coco_eval.params.imgIds = evaluated_imgs
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()

def eval_rnnt():
  # RNN-T
  from models.rnnt import RNNT
  mdl = RNNT()
  mdl.load_from_pretrained()

  from datasets.librispeech import iterate
  from examples.mlperf.metrics import word_error_rate

  LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

  c = 0
  scores = 0
  words = 0
  st = time.perf_counter()
  for X, Y in iterate():
    mt = time.perf_counter()
    tt = mdl.decode(Tensor(X[0]), Tensor([X[1]]))
    et = time.perf_counter()
    print(f"{(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model")
    for n, t in enumerate(tt):
      tnp = np.array(t)
      _, scores_, words_ = word_error_rate(["".join([LABELS[int(tnp[i])] for i in range(tnp.shape[0])])], [Y[n]])
      scores += scores_
      words += words_
    c += len(tt)
    print(f"WER: {scores/words}, {words} words, raw scores: {scores}, c: {c}")
    st = time.perf_counter()

def eval_bert():
  # Bert-QA
  from models.bert import BertForQuestionAnswering
  mdl = BertForQuestionAnswering()
  mdl.load_from_pretrained()

  @TinyJit
  def run(input_ids, input_mask, segment_ids):
    return mdl(input_ids, input_mask, segment_ids).realize()

  from datasets.squad import iterate
  from examples.mlperf.helpers import get_bert_qa_prediction
  from examples.mlperf.metrics import f1_score
  from transformers import BertTokenizer

  tokenizer = BertTokenizer(str(Path(__file__).parent.parent.parent / "weights/bert_vocab.txt"))

  c = 0
  f1 = 0.0
  st = time.perf_counter()
  for X, Y in iterate(tokenizer):
    mt = time.perf_counter()
    outs = []
    for x in X:
      outs.append(run(Tensor(x["input_ids"]), Tensor(x["input_mask"]), Tensor(x["segment_ids"])).numpy())
    et = time.perf_counter()
    print(f"{(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model over {len(X)} features")

    pred = get_bert_qa_prediction(X, Y, outs)
    print(f"pred: {pred}\nans: {Y['answers']}")
    f1 += max([f1_score(pred, ans) for ans in Y["answers"]])
    c += 1
    print(f"f1: {f1/c}, raw: {f1}, c: {c}\n")

    st = time.perf_counter()

if __name__ == "__main__":
  # inference only
  Tensor.training = False
  Tensor.no_grad = True

  models = getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert").split(",")
  for m in models:
    nm = f"eval_{m}"
    if nm in globals():
      print(f"eval {m}")
      globals()[nm]()
