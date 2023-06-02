from tinygrad.tensor import Tensor,Device
from tinygrad.helpers import getenv
from tinygrad.nn import optim
from tqdm import tqdm

def train_resnet():
  # TODO: Resnet50-v1.5
  # there's no weight decay for sgd ?
  from models.resnet import ResNet50
  from datasets.imagenet import iterate,get_train_files
  from extra.lr_scheduler import CosineAnnealingLR
  from examples.mlperf.metrics import cross_entropy_loss
  from extra.training import sparse_categorical_crossentropy

  classes = 100
  model = ResNet50(classes)
  parameters = optim.get_parameters(model)
  BS = 8
  lr = BS*1e-3
  epochs = 50
  optimizer = optim.SGD(parameters, lr=lr, momentum=.875)
  scheduler = CosineAnnealingLR(optimizer, epochs)
  print(f"training with batch size {BS} for {epochs} epochs")
  def warmup_factor(step, epochs):
    return min(1.0, (step+1)/epochs)

  Tensor.training = True
  for epoch in (r := tqdm(range(epochs))):
    losses,accs = 0,0
    for X,Y in (t := tqdm(iterate(bs=BS, val=False), total=len(get_train_files())//4000//BS)):
      out = model.forward(Tensor(X, requires_grad=False))
      loss = sparse_categorical_crossentropy(out, Y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss = loss.cpu().numpy()
      cat = out.cpu().numpy().argmax(axis=1)
      accuracy = (cat == Y).mean()
      losses += loss//len(get_train_files())//BS
      accs += accuracy//len(get_train_files())//BS
      t.set_description("loss %.2f accuracy %.2f : lr %.3f" % (loss, accuracy, scheduler.get_lr()))
      del out, loss
    scheduler.step()
    r.set_description("loss %.2f accuracy %.2f : lr %.3f" % (losses, accs, scheduler.get_lr()))

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
  # TODO: BERT
  pass

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


