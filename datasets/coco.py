import json
import pathlib
import zipfile
import numpy as np
from tinygrad.tensor import Tensor
from extra.utils import download_file
import pycocotools._mask as _mask
from examples.mask_rcnn import Masker
from examples.mask_rcnn import transforms_train
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torchvision

from models.mask_rcnn import BoxList
from models.mask_rcnn import SegmentationMask
from models.mask_rcnn import PersonKeypoints

iou         = _mask.iou
merge       = _mask.merge
frPyObjects = _mask.frPyObjects

BASEDIR = pathlib.Path(__file__).parent.parent / "datasets/COCO"

def create_dict(key_row, val_row, rows): return {row[key_row]:row[val_row] for row in rows}

if not pathlib.Path(BASEDIR/'train2017').is_dir():
  fn = BASEDIR/'train2017.zip'
  download_file('http://images.cocodataset.org/zips/train2017.zip',fn)
  with zipfile.ZipFile(fn, 'r') as zip_ref:
    zip_ref.extractall(BASEDIR)
  fn.unlink()

if not pathlib.Path(BASEDIR/'val2017').is_dir():
  fn = BASEDIR/'val2017.zip'
  download_file('http://images.cocodataset.org/zips/val2017.zip',fn)
  with zipfile.ZipFile(fn, 'r') as zip_ref:
    zip_ref.extractall(BASEDIR)
  fn.unlink()
    
if not pathlib.Path(BASEDIR/'annotations').is_dir():
  fn = BASEDIR/'annotations_trainval2017.zip'
  download_file('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',fn)
  with zipfile.ZipFile(fn, 'r') as zip_ref:
    zip_ref.extractall(BASEDIR)
  fn.unlink()

# Training
min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
  return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
  return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
  # if it's empty, there is no annotation
  if len(anno) == 0:
    return False
  # if all boxes have close to zero area, there is no annotation
  if _has_only_empty_bbox(anno):
    return False
  # keypoints task have a slight different critera for considering
  # if an annotation is valid
  if "keypoints" not in anno[0]:
    return True
  # for keypoint detection tasks, only consider valid images those
  # containing at least min_keypoints_per_image
  if _count_visible_keypoints(anno) >= min_keypoints_per_image:
    return True
  return False

class COCODataset(torchvision.datasets.coco.CocoDetection):
  def __init__(self, root=pathlib.Path(BASEDIR/'train2017'),
                     ann_file=pathlib.Path(BASEDIR/'annotations/instances_train2017.json'), 
                     remove_images_without_annotations=1, 
                     transforms=transforms_train):

    super(COCODataset, self).__init__(root, ann_file)
    # sort indices for reproducible results
    self.ids = sorted(self.ids)

    # filter images without detection annotations
    if remove_images_without_annotations:
      ids = []
      for img_id in self.ids:
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = self.coco.loadAnns(ann_ids)
        if has_valid_annotation(anno):
          ids.append(img_id)
      self.ids = ids

    self.json_category_id_to_contiguous_id = {
        v: i + 1 for i, v in enumerate(self.coco.getCatIds())
    }
    self.contiguous_category_id_to_json_id = {
        v: k for k, v in self.json_category_id_to_contiguous_id.items()
    }
    self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
    self._transforms = transforms

  def __getitem__(self, idx):
    img, anno = super(COCODataset, self).__getitem__(idx)

    # filter crowd annotations
    # TODO might be better to add an extra field
    anno = [obj for obj in anno if obj["iscrowd"] == 0]

    boxes = [obj["bbox"] for obj in anno]
    boxes = Tensor(boxes).reshape(-1, 4)  # guard against no boxes
    target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

    classes = [obj["category_id"] for obj in anno]
    classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
    classes = Tensor(classes)
    target.add_field("labels", classes)

    masks = [obj["segmentation"] for obj in anno]
    masks = SegmentationMask(masks, img.size)
    target.add_field("masks", masks)

    if anno and "keypoints" in anno[0]:
      keypoints = [obj["keypoints"] for obj in anno]
      keypoints = PersonKeypoints(keypoints, img.size)
      target.add_field("keypoints", keypoints)

    target = target.clip_to_image(remove_empty=True)

    if self._transforms is not None:
      img, target = self._transforms(img, target)

    return img, target, idx

  def get_img_info(self, index):
    img_id = self.id_to_img_map[index]
    img_data = self.coco.imgs[img_id]
    return img_data

# Inference
with open(BASEDIR/'annotations/instances_val2017.json', 'r') as f:
  annotations_raw = json.loads(f.read())
images = annotations_raw['images']
categories = annotations_raw['categories']
annotations = annotations_raw['annotations']
file_name_to_id = create_dict('file_name', 'id', images)
id_to_width = create_dict('id', 'width', images)
id_to_height = create_dict('id', 'height', images)
json_category_id_to_contiguous_id = {v['id']: i + 1 for i, v in enumerate(categories)}
contiguous_category_id_to_json_id = {v:k for k,v in json_category_id_to_contiguous_id.items()}

def encode(bimask):
  if len(bimask.shape) == 3:
    return _mask.encode(bimask)
  elif len(bimask.shape) == 2:
    h, w = bimask.shape
    return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]

def decode(rleObjs):
  if type(rleObjs) == list:
    return _mask.decode(rleObjs)
  else:
    return _mask.decode([rleObjs])[:,:,0]

def area(rleObjs):
  if type(rleObjs) == list:
    return _mask.area(rleObjs)
  else:
    return _mask.area([rleObjs])[0]

def toBbox(rleObjs):
  if type(rleObjs) == list:
    return _mask.toBbox(rleObjs)
  else:
    return _mask.toBbox([rleObjs])[0]


def convert_prediction_to_coco_bbox(file_name, prediction):
  coco_results = []
  try:
    original_id = file_name_to_id[file_name]
    if len(prediction) == 0:
      return coco_results

    image_width = id_to_width[original_id]
    image_height = id_to_height[original_id]
    prediction = prediction.resize((image_width, image_height))
    prediction = prediction.convert("xywh")

    boxes = prediction.bbox.numpy().tolist()
    scores = prediction.get_field("scores").numpy().tolist()
    labels = prediction.get_field("labels").numpy().tolist()

    mapped_labels = [contiguous_category_id_to_json_id[int(i)] for i in labels]

    coco_results.extend(
      [
        {
          "image_id": original_id,
          "category_id": mapped_labels[k],
          "bbox": box,
          "score": scores[k],
        }
          for k, box in enumerate(boxes)
      ]
    )
  except Exception as e:
    print(file_name, e)
  return coco_results

masker = Masker(threshold=0.5, padding=1)

def convert_prediction_to_coco_mask(file_name, prediction):
  coco_results = []
  try:
    original_id = file_name_to_id[file_name]
    if len(prediction) == 0:
      return coco_results

    image_width = id_to_width[original_id]
    image_height = id_to_height[original_id]
    prediction = prediction.resize((image_width, image_height))
    masks = prediction.get_field("mask")

    scores = prediction.get_field("scores").numpy().tolist()
    labels = prediction.get_field("labels").numpy().tolist()

    masks = masker([masks], [prediction])[0].numpy()

    rles = [
      encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
      for mask in masks
    ]
    for rle in rles:
      rle["counts"] = rle["counts"].decode("utf-8")

    mapped_labels = [contiguous_category_id_to_json_id[int(i)] for i in labels]

    coco_results.extend(
      [
        {
          "image_id": original_id,
          "category_id": mapped_labels[k],
          "segmentation": rle,
          "score": scores[k],
        }
          for k, rle in enumerate(rles)
      ]
    )
  except Exception as e:
    print(file_name, e)
  return coco_results

def accumulate_predictions_for_coco(coco_results, json_result_file, rm=False):
  path = pathlib.Path(json_result_file)
  if rm and path.exists(): path.unlink()
  with open(path, "a") as f:
    for s in coco_results:
      f.write(json.dumps(s))
      f.write('\n')

def remove_dup(l):
  seen = set()
  seen_add = seen.add
  return [x for x in l if not (x in seen or seen_add(x))]

class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super(NpEncoder, self).default(obj)


def evaluate_predictions_on_coco(json_result_file, iou_type="bbox"):
  coco_results = []
  with open(json_result_file, "r") as f:
    for line in f:
      coco_results.append(json.loads(line))
  
  coco_gt = COCO(str(BASEDIR/'annotations/instances_val2017.json'))
  set_of_json = remove_dup([json.dumps(d, cls=NpEncoder) for d in coco_results])
  unique_list = [json.loads(s) for s in set_of_json]

  with open(f'{json_result_file}.flattend', "w") as f:
    json.dump(unique_list, f)

  coco_dt = coco_gt.loadRes(str(f'{json_result_file}.flattend')) 
  coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  return coco_eval

def iterate(files, bs=1):
  batch = []
  for file in files:
    batch.append(file)
    if len(batch) >= bs: yield batch; batch = []
  if len(batch) > 0: yield batch; batch = []
