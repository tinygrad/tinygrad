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
  
  from extra.datasets.coco import COCODataset
  from extra.datasets.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize
  from models.mask_rcnn import MaskRCNN
  from models.resnet import ResNet

  resnet = ResNet(50, num_classes=None, stride_in_1x1=True)
  model = MaskRCNN(backbone=resnet)
  model.load_from_pretrained()

  transform = Compose(
    [
      Resize(800, 1333),
      RandomHorizontalFlip(0.5),
      ToTensor(),
      Normalize(
        mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.], to_bgr255=True
      ),
    ]
  )

  dataset = COCODataset(root='extra/datasets/COCO/train2017',
                        ann_file='extra/datasets/COCO/annotations/instances_train2017.json', 
                        remove_images_without_annotations=1, 
                        transforms=transform)

  # Sanity test while refactoring the COCO dataset code base
  # print("Data")
  # print(dataset[0][0])
  print("Bounding box")
  print(dataset[0][1].bbox[0])
  print(dataset[0][1].bbox[0].numpy())
  print("Label")
  print(dataset[0][1].get_field('labels')[0])
  print(dataset[0][1].get_field('labels')[0].numpy())
  print("Segmentation mask")
  # print(dataset[0][1].get_field('masks'))
  # print(dataset[0][1].get_field('masks').polygons)
  # print(dataset[0][1].get_field('masks').polygons[0])
  print(dataset[0][1].get_field('masks').polygons[0].polygons[0])
  print(dataset[0][1].get_field('masks').polygons[0].polygons[0].numpy())
  # for p in dataset[0][1].get_field('masks').polygons:
  # print(p.polygons)
  #   for p2 in p.polygons:
  #     print(p2)

  # For training, you must also adjust the learning rate and schedule length 
  # according to the linear scaling rule. See for example:
  # https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14
  # optimizer = optim.SGD(optim.get_parameters(model), lr=0.0025, weight_decay=0.0005, momentum=0.9)

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


