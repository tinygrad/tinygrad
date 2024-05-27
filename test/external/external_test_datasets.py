from extra.datasets.kits19 import iterate, preprocess
from test.external.mlperf_unet3d.kits19 import PytTrain, PytVal
from tinygrad.helpers import temp
from pathlib import Path

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

  def _create_sample(self, val):
    self._set_seed()

    img, lbl = np.random.rand(190, 392, 392).astype(np.float32), np.random.randint(0, 100, size=(190, 392, 392)).astype(np.uint8)
    img, lbl = nib.Nifti1Image(img, np.eye(4)), nib.Nifti1Image(lbl, np.eye(4))

    os.makedirs(tempfile.gettempdir() + "/case_0000", exist_ok=True)
    nib.save(img, temp("case_0000/imaging.nii.gz"))
    nib.save(lbl, temp("case_0000/segmentation.nii.gz"))

    sample_pth = Path(tempfile.gettempdir()) / "case_0000"
    img, lbl = preprocess(sample_pth)
    dataset = "val" if val else "train"
    preproc_img_pth, preproc_lbl_pth = temp(f"{dataset}/case_0000_x.npy"), temp(f"{dataset}/case_0000_y.npy")

    os.makedirs(tempfile.gettempdir() + f"/{dataset}", exist_ok=True)
    np.save(preproc_img_pth, img, allow_pickle=False)
    np.save(preproc_lbl_pth, lbl, allow_pickle=False)

    return Path(tempfile.gettempdir()) / dataset, sample_pth #TODO: remove sample_pth afterwards once it is moved to dataloader

  def _create_kits19_ref_sample(self, preprocessed_dataset_dir, val):
    self._set_seed()

    preproc_img_pth, preproc_lbl_pth = preprocessed_dataset_dir / "case_0000_x.npy", preprocessed_dataset_dir / "case_0000_y.npy"

    if val:
      dataset = PytVal([preproc_img_pth], [preproc_lbl_pth])
    else:
      dataset = PytTrain([preproc_img_pth], [preproc_lbl_pth], patch_size=(128, 128, 128), oversampling=0.4)

    return dataset[0]

  def _create_kits19_tinygrad_sample(self, sample_pth, val):
    self._set_seed()
    return next(iterate([sample_pth], val=val))

  def test_kits19_training_set(self):
    preprocessed_dataset_dir, sample_pth = self._create_sample(False)

    tinygrad_img, tinygrad_lbl = self._create_kits19_tinygrad_sample(sample_pth, False)
    ref_img, ref_lbl = self._create_kits19_ref_sample(preprocessed_dataset_dir, False)

    np.testing.assert_equal(tinygrad_img[:, 0], ref_img)
    np.testing.assert_equal(tinygrad_lbl[:, 0], ref_lbl)

  def test_kits19_validation_set(self):
    preprocessed_dataset_dir, sample_pth = self._create_sample(True)

    tinygrad_img, tinygrad_lbl = self._create_kits19_tinygrad_sample(sample_pth, True)
    ref_img, ref_lbl = self._create_kits19_ref_sample(preprocessed_dataset_dir, True)

    np.testing.assert_equal(tinygrad_img[:, 0], ref_img)
    np.testing.assert_equal(tinygrad_lbl, ref_lbl)

if __name__ == '__main__':
  unittest.main()
