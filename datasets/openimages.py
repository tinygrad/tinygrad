import math
import json
from extra.utils import OSX
import numpy as np
from PIL import Image
import pathlib

BASEDIR = pathlib.Path(__file__).parent.parent / "datasets/open-images-v6-mlperf"
MLPERF_CLASSES = ['Airplane', 'Antelope', 'Apple', 'Backpack', 'Balloon', 'Banana',
  'Barrel', 'Baseball bat', 'Baseball glove', 'Bee', 'Beer', 'Bench', 'Bicycle',
  'Bicycle helmet', 'Bicycle wheel', 'Billboard', 'Book', 'Bookcase', 'Boot',
  'Bottle', 'Bowl', 'Bowling equipment', 'Box', 'Boy', 'Brassiere', 'Bread',
  'Broccoli', 'Bronze sculpture', 'Bull', 'Bus', 'Bust', 'Butterfly', 'Cabinetry',
  'Cake', 'Camel', 'Camera', 'Candle', 'Candy', 'Cannon', 'Canoe', 'Carrot', 'Cart',
  'Castle', 'Cat', 'Cattle', 'Cello', 'Chair', 'Cheese', 'Chest of drawers', 'Chicken',
  'Christmas tree', 'Coat', 'Cocktail', 'Coffee', 'Coffee cup', 'Coffee table', 'Coin',
  'Common sunflower', 'Computer keyboard', 'Computer monitor', 'Convenience store',
  'Cookie', 'Countertop', 'Cowboy hat', 'Crab', 'Crocodile', 'Cucumber', 'Cupboard',
  'Curtain', 'Deer', 'Desk', 'Dinosaur', 'Dog', 'Doll', 'Dolphin', 'Door', 'Dragonfly',
  'Drawer', 'Dress', 'Drum', 'Duck', 'Eagle', 'Earrings', 'Egg (Food)', 'Elephant',
  'Falcon', 'Fedora', 'Flag', 'Flowerpot', 'Football', 'Football helmet', 'Fork',
  'Fountain', 'French fries', 'French horn', 'Frog', 'Giraffe', 'Girl', 'Glasses',
  'Goat', 'Goggles', 'Goldfish', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Guitar',
  'Hamburger', 'Handbag', 'Harbor seal', 'Headphones', 'Helicopter', 'High heels',
  'Hiking equipment', 'Horse', 'House', 'Houseplant', 'Human arm', 'Human beard',
  'Human body', 'Human ear', 'Human eye', 'Human face', 'Human foot', 'Human hair',
  'Human hand', 'Human head', 'Human leg', 'Human mouth', 'Human nose', 'Ice cream',
  'Jacket', 'Jeans', 'Jellyfish', 'Juice', 'Kitchen & dining room table', 'Kite',
  'Lamp', 'Lantern', 'Laptop', 'Lavender (Plant)', 'Lemon', 'Light bulb', 'Lighthouse',
  'Lily', 'Lion', 'Lipstick', 'Lizard', 'Man', 'Maple', 'Microphone', 'Mirror',
  'Mixing bowl', 'Mobile phone', 'Monkey', 'Motorcycle', 'Muffin', 'Mug', 'Mule',
  'Mushroom', 'Musical keyboard', 'Necklace', 'Nightstand', 'Office building',
  'Orange', 'Owl', 'Oyster', 'Paddle', 'Palm tree', 'Parachute', 'Parrot', 'Pen',
  'Penguin', 'Personal flotation device', 'Piano', 'Picture frame', 'Pig', 'Pillow',
  'Pizza', 'Plate', 'Platter', 'Porch', 'Poster', 'Pumpkin', 'Rabbit', 'Rifle',
  'Roller skates', 'Rose', 'Salad', 'Sandal', 'Saucer', 'Saxophone', 'Scarf', 'Sea lion',
  'Sea turtle', 'Sheep', 'Shelf', 'Shirt', 'Shorts', 'Shrimp', 'Sink', 'Skateboard',
  'Ski', 'Skull', 'Skyscraper', 'Snake', 'Sock', 'Sofa bed', 'Sparrow', 'Spider', 'Spoon',
  'Sports uniform', 'Squirrel', 'Stairs', 'Stool', 'Strawberry', 'Street light',
  'Studio couch', 'Suit', 'Sun hat', 'Sunglasses', 'Surfboard', 'Sushi', 'Swan',
  'Swimming pool', 'Swimwear', 'Tank', 'Tap', 'Taxi', 'Tea', 'Teddy bear', 'Television',
  'Tent', 'Tie', 'Tiger', 'Tin can', 'Tire', 'Toilet', 'Tomato', 'Tortoise', 'Tower',
  'Traffic light', 'Train', 'Tripod', 'Truck', 'Trumpet', 'Umbrella', 'Van', 'Vase',
  'Vehicle registration plate', 'Violin', 'Wall clock', 'Waste container', 'Watch',
  'Whale', 'Wheel', 'Wheelchair', 'Whiteboard', 'Window', 'Wine', 'Wine glass', 'Woman',
  'Zebra', 'Zucchini',
]

def openimages():
  ann_file = BASEDIR / "validation/labels/openimages-mlperf.json"
  if not ann_file.is_file():
    fetch_openimages(ann_file)
  return ann_file

def fetch_openimages(output_fname):
  import fiftyone as fo
  import fiftyone.zoo as foz
  dataset = foz.load_zoo_dataset("open-images-v6", classes=MLPERF_CLASSES, splits="validation", label_types="detections", dataset_dir=BASEDIR)
  print("Converting dataset to coco format ...")
  dataset.export(labels_path=output_fname, dataset_type=fo.types.COCODetectionDataset, classes=MLPERF_CLASSES)

  # Add iscrowd label to openimages annotations
  with open(output_fname) as fp:
    labels = json.load(fp)
  for annotation in labels['annotations']:
    annotation['iscrowd'] = int(annotation['IsGroupOf'])
  with open(output_fname, "w") as fp:
    json.dump(labels, fp)

def image_load(fn):
  img_folder = BASEDIR / "validation/data"
  img = Image.open(img_folder / fn).convert('RGB')
  import torchvision.transforms.functional as F
  ret = F.resize(img, size=(800, 800))
  ret = np.array(ret)
  return ret, img.size[::-1]

def prepare_target(annotations, img_id, img_size):
  boxes = [annot["bbox"] for annot in annotations]
  boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
  boxes[:, 2:] += boxes[:, :2]
  boxes[:, 0::2] = boxes[:, 0::2].clip(0, img_size[1])
  boxes[:, 1::2] = boxes[:, 1::2].clip(0, img_size[0])
  keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
  boxes = boxes[keep]
  classes = [annot["category_id"] for annot in annotations]
  classes = np.array(classes, dtype=np.int64)
  classes = classes[keep]
  return {"boxes": boxes, "labels": classes, "image_id": img_id, "image_size": img_size}

def iterate(coco, bs=16):
  image_ids = sorted(coco.imgs.keys())
  for i in range(0, len(image_ids), bs):
    X, targets  = [], []
    for img_id in image_ids[i:i+bs]:
      x, original_size = image_load(coco.loadImgs(img_id)[0]["file_name"])
      X.append(x)
      annotations = coco.loadAnns(coco.getAnnIds(img_id))
      targets.append(prepare_target(annotations, img_id, original_size))
    yield np.array(X), targets
