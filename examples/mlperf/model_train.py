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
  # TODO: BERT
  pass

def train_maskrcnn():
  from datasets.coco import COCODataset
  from models.mask_rcnn import MaskRCNN
  from models.resnet import ResNet

  resnet = ResNet(50, num_classes=None, stride_in_1x1=True)
  model = MaskRCNN(backbone=resnet)
  model.load_from_pretrained()
  # For training, you must also adjust the learning rate and schedule length 
  # according to the linear scaling rule. See for example:
  # https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14
  optimizer = optim.SGD(optim.get_parameters(model), lr=0.0025, weight_decay=0.0005, momentum=0.9)

  dataset = COCODataset()

  # scheduler = make_lr_scheduler(cfg, optimizer)

  # data_loader, iters_per_epoch = make_data_loader(
      # cfg,
      # is_train=True,
      # is_distributed=distributed,
      # start_iter=arguments["iteration"],
      # random_number_generator=random_number_generator
  # )

  # The mlcommons has the loss functions defined inside their model definition
  #
  # loss_dict = model(images, targets)
  # losses = sum(loss for loss in loss_dict.values()) 
  # 
  # do_train(
  #     model,
  #     data_loader,
  #     optimizer,
  #     scheduler,
  #     checkpointer,
  #     device,
  #     checkpoint_period,
  #     arguments,
  # )

if __name__ == "__main__":
  Tensor.training = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
    nm = f"train_{m}"
    if nm in globals():
      print(f"training {m}")
      globals()[nm]()


