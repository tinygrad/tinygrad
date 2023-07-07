import pathlib
import zipfile
import numpy as np
import torchvision
import pycocotools._mask as _mask

from tinygrad.tensor import Tensor
from extra.utils import download_file
from models.mask_rcnn import BoxList
from models.mask_rcnn import SegmentationMask
from models.mask_rcnn import PersonKeypoints


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

# Note:
# torch does image and target transform inside VisionDataset base class while 
# here it is done outside of the class
class COCODataset(torchvision.datasets.coco.CocoDetection):
  def __init__(self, root='COCO/train2017',
                     ann_file='COCO/annotations/instances_train2017.json', 
                     remove_images_without_annotations=1, 
                     transforms=None):

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
    print(img.size)
    print(np.shape(np.array(img)))
    # img = np.array(img)
    # Raw image before any opration in case I need to check my sanity
    # import matplotlib.pyplot as plt
    # plt.imshow(np.array(img))
    # plt.savefig('/home/iris/yg5d6/Workspace/coco.png')

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

if __name__=="__main__":
  
  BASEDIR = pathlib.Path(__file__).parent.parent / "datasets" / "COCO"
  BASEDIR.mkdir(exist_ok=True)

  # Download data and annotations
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