from extra.datasets.kits19 import iterate, preprocess
from test.external.mlperf_unet3d.kits19 import PytTrain, PytVal
from tinygrad import Tensor
from tinygrad.helpers import temp
from pathlib import Path

import nibabel as nib
import numpy as np
import os
import random
import tempfile
import unittest

class ExternalTestDatasets(unittest.TestCase):
  def setUp(self):
    Tensor.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

  def _create_sample(self):
    img, lbl = np.random.rand(190, 392, 392).astype(np.float32), np.ones((190, 392, 392)).astype(np.uint8)
    img, lbl = nib.Nifti1Image(img, np.eye(4)), nib.Nifti1Image(lbl, np.eye(4))

    os.makedirs(tempfile.gettempdir() + "case_0000", exist_ok=True)
    nib.save(img, temp("case_0000/imaging.nii.gz"))
    nib.save(lbl, temp("case_0000/segmentation.nii.gz"))

    return Path(tempfile.gettempdir()) / "case_0000"

  def _preprocess_sample(self, sample_pth, val):
    img, lbl = preprocess(sample_pth)
    dataset = "val" if val else "train"
    img_path, lbl_path = temp(f"{dataset}/case_0000_x.npy"), temp(f"{dataset}/case_0000_y.npy")

    os.makedirs(tempfile.gettempdir() + f"/{dataset}", exist_ok=True)
    np.save(img_path, img, allow_pickle=False)
    np.save(lbl_path, lbl, allow_pickle=False)

    return img_path, lbl_path

  def test_kits19_training_set(self):
    sample_pth = self._create_sample()
    tinygrad_img, tinygrad_lbl = next(iterate([sample_pth], val=False))
    preprocessed_img_pth, preprocessed_lbl_pth = self._preprocess_sample(sample_pth, False)

    ref_dataset = PytTrain([preprocessed_img_pth], [preprocessed_lbl_pth], patch_size=(128, 128, 128), oversampling=0.4)
    ref_img, ref_lbl = ref_dataset[0]

    np.testing.assert_allclose(tinygrad_img[:, 0, :, :, :], ref_img, atol=1e-1)
    np.testing.assert_equal(tinygrad_lbl[:, 0, :, :, :], ref_lbl)

  def test_kits19_validation_set(self):
    sample_pth = self._create_sample()
    tinygrad_img, tinygrad_lbl = next(iterate([sample_pth], val=True))
    preprocessed_img_pth, preprocessed_lbl_pth = self._preprocess_sample(sample_pth, False)

    ref_dataset = PytVal([preprocessed_img_pth], [preprocessed_lbl_pth])
    ref_img, ref_lbl = ref_dataset[0]

    np.testing.assert_allclose(tinygrad_img[:, 0, :, :, :], ref_img, atol=1e-1)
    np.testing.assert_equal(tinygrad_lbl, ref_lbl)

if __name__ == '__main__':
  unittest.main()
