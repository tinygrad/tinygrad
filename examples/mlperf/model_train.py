from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, Context


def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  from models.resnet import ResNeXt50_32X4D
  from models.retinanet import RetinaNet
  from examples.mlperf.train_retinanet import RetinaNetTrainer
  backbone = ResNeXt50_32X4D()
  retina = RetinaNet(backbone) #remember num_classes = 600 for openimages
  trainer = RetinaNetTrainer(retina)
  trainer.train()
 


def example_inference(np, fo, os, input_fixup, retina):
    #TODO delete this once training loop works fine
    images = load_openimages_as_tg_tensor(np, fo, os, input_fixup)
    images = input_fixup(images)

    model_detection_embeds = retina(images).numpy()
    model_detections = retina.postprocess_detections(model_detection_embeds)
    return model_detections

def load_openimages_as_tg_tensor(np, fo, os, input_fixup, n_images=24, img_reshape = (100,100)):
    from PIL import Image
    Warning("Enhance with DataLoader") #TODO
    #train_16_batch = fo.zoo.load_zoo_dataset("open-images-v6",split="train",label_types="detections", max_samples=n_images)
    #classes = train_16_batch.distinct("ground_truth.detections.label")
    dir_example  = r'C:\Users\msoro\fiftyone\open-images-v6\train\data\00001bc2c4027449.jpg'
    #IMAGES_DIR = os.path.dirname(train_16_batch.first().filepath)
    IMAGES_DIR = os.path.dirname(dir_example)
    
    #IMAGES_DIR = os.path.dirname(r"C:\Users\msoro\fiftyone\open-images-v6\train\data\000002b66c9c498e.jpg'")
    """train_16_batch.take(n_images).export(
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
    labels_path="/tmp/coco.json",
    classes=classes,)
  """
    coco_dataset = fo.Dataset.from_dir(dataset_type=fo.types.COCODetectionDataset,data_path=IMAGES_DIR,labels_path="/tmp/coco.json",include_id=True, n_images=n_images)
    #tiny images for debugging...
    #images = [Image.open(sample.filepath).resize(img_reshape) for sample in coco_dataset]
    #images = Tensor([np.asarray(im) for im in images])
    return coco_dataset
  #REMEMBER: Quality target = 34.0% mAP
  #TODO 3: adapt for mlperf-retinanet training standard 
  # reference torch implementation https://github.com/mlcommons/training/blob/master/object_detection/pytorch/maskrcnn_benchmark/modeling/rpn/retinanet


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


