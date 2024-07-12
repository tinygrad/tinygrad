import glob, random, json, math, os,sys
import numpy as np
from PIL import Image
import functools, pathlib
from tinygrad.helpers import diskcache, getenv

BASEDIR = pathlib.Path(__file__).parent / "open-images-v6-mlperf"

# @diskcache
def get_train_files():
  if not (files:=glob.glob(p:=str(BASEDIR / "train/data/*"))): raise FileNotFoundError(f"No training files in {p}")
  return files
def get_train_data():
  with open(os.path.join(BASEDIR, 'train','train_data.json')) as f:
    data = json.load(f)
  return data
def get_val_data():
  with open(os.path.join(BASEDIR, 'validation', 'labels','openimages-mlperf.json')) as f:
    data = json.load(f)
  return data
# @functools.lru_cache(None)
def get_val_files():
  if not (files:=glob.glob(p:=str(BASEDIR / "validation/data/*"))): raise FileNotFoundError(f"No validation files in {p}")
  return files


def rand_flip(img):
  if random.random() < 0.5:
    img = np.flip(img, axis=1).copy()
  return img

def img_resize_convert(img, size):
  return img.resize((size, size), resample = Image.BILINEAR)

def preprocess_train(img):
  img = np.array(img_resize_convert(img, 800))
  # print(img.shape)
  # sys.exit()
  # img = rand_flip(img)
  return img

def box_iou_np(boxes1:np.ndarray, boxes2:np.ndarray):
  def box_area(boxes): return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)
  
  lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
  rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
  wh = (rb-lt).clip(min=0)
  inter = wh[:, :, 0] * wh[:, :, 1]
  union = area1[:, None] + area2 - inter
  return inter / union

def matcher_np(match_quality_matrix:np.ndarray):
  assert match_quality_matrix.size != 0, 'Need valid matrix'

  def set_low_quality_matches_(matches:np.ndarray, all_matches:np.ndarray, match_quality_matrix:np.ndarray):

    highest_quality_foreach_gt= match_quality_matrix.max(axis=1)
    gt_pred_pairs_of_highest_quality = np.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])
    pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
    matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

  # match_quality_matrix is M (gt) x N (predicted)
  # Max over gt elements (dim 0) to find best gt candidate for each prediction
  matched_vals, matches = match_quality_matrix.max(axis=0), match_quality_matrix.argmax(axis=0)
  all_matches = np.copy(matches)

  # Assign candidate matches with low quality to negative (unassigned) values
  below_low_threshold = matched_vals < 0.4
  between_thresholds = (matched_vals >= 0.4) & (matched_vals < 0.5)
  matches[below_low_threshold] = -1
  matches[between_thresholds] = -2

  assert all_matches is not None
  set_low_quality_matches_(matches, all_matches, match_quality_matrix)
  return matches

def matcher_iou_func(box, anchor): return matcher_np(box_iou_np(box, anchor))
