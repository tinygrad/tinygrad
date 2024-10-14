from extra.datasets.kits19 import iterate, preprocess
from examples.mlperf.dataloader import batch_load_unet3d
from test.external.mlperf_retinanet.openimages import get_openimages
from test.external.mlperf_unet3d.kits19 import PytTrain, PytVal
from tinygrad.helpers import temp
from pathlib import Path
from PIL import Image

import json
import nibabel as nib
import numpy as np
import os
import random
import tempfile
import unittest

class ExternalTestDatasets(unittest.TestCase):
  def _set_seed(self):
    np.random.seed(42)
    random.seed(42)

class TestKiTS19Dataset(ExternalTestDatasets):
  def _create_samples(self, val, num_samples=2):
    self._set_seed()

    img, lbl = np.random.rand(190, 392, 392).astype(np.float32), np.random.randint(0, 100, size=(190, 392, 392)).astype(np.uint8)
    img, lbl = nib.Nifti1Image(img, np.eye(4)), nib.Nifti1Image(lbl, np.eye(4))
    dataset = "val" if val else "train"
    preproc_pth = Path(tempfile.gettempdir() + f"/{dataset}")

    for i in range(num_samples):
      os.makedirs(tempfile.gettempdir() + f"/case_000{i}", exist_ok=True)
      nib.save(img, temp(f"case_000{i}/imaging.nii.gz"))
      nib.save(lbl, temp(f"case_000{i}/segmentation.nii.gz"))

      preproc_img, preproc_lbl = preprocess(Path(tempfile.gettempdir()) / f"case_000{i}")
      preproc_img_pth, preproc_lbl_pth = temp(f"{dataset}/case_000{i}_x.npy"), temp(f"{dataset}/case_000{i}_y.npy")

      os.makedirs(preproc_pth, exist_ok=True)
      np.save(preproc_img_pth, preproc_img, allow_pickle=False)
      np.save(preproc_lbl_pth, preproc_lbl, allow_pickle=False)

    return preproc_pth, list(preproc_pth.glob("*_x.npy")), list(preproc_pth.glob("*_y.npy"))

  def _create_ref_dataloader(self, preproc_img_pths, preproc_lbl_pths, val):
    if val:
      dataset = PytVal(preproc_img_pths, preproc_lbl_pths)
    else:
      dataset = PytTrain(preproc_img_pths, preproc_lbl_pths, patch_size=(128, 128, 128), oversampling=0.4)

    return iter(dataset)

  def _create_tinygrad_dataloader(self, preproc_pth, val, batch_size=1, shuffle=False, seed=42, use_old_dataloader=False):
    if use_old_dataloader:
      dataset = iterate(list(Path(tempfile.gettempdir()).glob("case_*")), preprocessed_dir=preproc_pth, val=val, shuffle=shuffle, bs=batch_size)
    else:
      dataset = iter(batch_load_unet3d(preproc_pth, batch_size=batch_size, val=val, shuffle=shuffle, seed=seed))

    return iter(dataset)

  def test_training_set(self):
    preproc_pth, preproc_img_pths, preproc_lbl_pths = self._create_samples(False)
    ref_dataset = self._create_ref_dataloader(preproc_img_pths, preproc_lbl_pths, False)
    tinygrad_dataset = self._create_tinygrad_dataloader(preproc_pth, False)

    for ref_sample, tinygrad_sample in zip(ref_dataset, tinygrad_dataset):
      self._set_seed()

      np.testing.assert_equal(tinygrad_sample[0][:, 0].numpy(), ref_sample[0])
      np.testing.assert_equal(tinygrad_sample[1][:, 0].numpy(), ref_sample[1])

  def test_validation_set(self):
    preproc_pth, preproc_img_pths, preproc_lbl_pths = self._create_samples(True)
    ref_dataset = self._create_ref_dataloader(preproc_img_pths, preproc_lbl_pths, True)
    tinygrad_dataset = self._create_tinygrad_dataloader(preproc_pth, True, use_old_dataloader=True)

    for ref_sample, tinygrad_sample in zip(ref_dataset, tinygrad_dataset):
      np.testing.assert_equal(tinygrad_sample[0][:, 0], ref_sample[0])
      np.testing.assert_equal(tinygrad_sample[1], ref_sample[1])

class TestOpenImagesDataset(ExternalTestDatasets):
  def _create_samples(self, subset):
    os.makedirs(Path(base_dir:=tempfile.gettempdir() + "/openimages") / f"{subset}/data", exist_ok=True)
    os.makedirs(base_dir / Path(f"{subset}/labels"), exist_ok=True)

    lbls, img_size = ["class_1", "class_2"], (447, 1024)
    cats = [{"id": i, "name": c, "supercategory": None} for i, c in enumerate(lbls)]
    imgs = [{"id": i, "file_name": f"image_{i}.jpg", "height": img_size[0], "width": img_size[1], "subset": subset, "license": None, "coco_url": None} for i in range(len(lbls))]
    annots = [
      {"id": i, "image_id": i, "category_id": 0, "bbox": [23.217183744, 31.75409775, 964.1241282560001, 326.09017434000003], "area": 314391.4050683996, "IsOccluded": 0, "IsInside": 0, "IsDepiction": 0, "IsTruncated": 0, "IsGroupOf": 0, "iscrowd": 0}
      for i in range(len(lbls))
    ]
    info = {"dataset": "openimages_mlperf", "version": "v6"}
    coco_annotations = {"info": info, "licenses": [], "categories": cats, "images": imgs, "annotations": annots}

    with open(ann_file:=base_dir / Path(f"{subset}/labels/openimages-mlperf.json"), "w") as fp:
      json.dump(coco_annotations, fp)

    for i in range(len(lbls)):
      img = Image.new("RGB", img_size[::-1])
      img.save(base_dir / Path(f"{subset}/data/image_{i}.jpg"))

    return base_dir, ann_file

  def _create_ref_dataloader(self, subset):
    base_dir, ann_file = self._create_samples(subset)
    print(f"{base_dir=} {ann_file=}")

  def _create_tinygrad_dataloader(self):
    pass

  def test_training_set(self):
    self._create_ref_dataloader("train")
    assert 1==0

if __name__ == '__main__':
  unittest.main()
