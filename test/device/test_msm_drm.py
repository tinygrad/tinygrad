import ctypes, unittest

from tinygrad.runtime.autogen import msm_drm


class TestMSMDRMUAPI(unittest.TestCase):
  def test_struct_layouts(self):
    layouts = {
      msm_drm.struct_drm_msm_timespec: (16, (0, 8)),
      msm_drm.struct_drm_msm_param: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_new: (16, (0, 8, 12)),
      msm_drm.struct_drm_msm_gem_info: (24, (0, 4, 8, 16, 20)),
      msm_drm.struct_drm_msm_gem_submit_cmd: (32, (0, 4, 8, 12, 16, 20, 24, 24)),
      msm_drm.struct_drm_msm_gem_submit_bo: (16, (0, 4, 8)),
      msm_drm.struct_drm_msm_gem_submit: (72, (0, 4, 8, 12, 16, 24, 32, 36, 40, 48, 56, 60, 64, 68)),
      msm_drm.struct_drm_msm_wait_fence: (32, (0, 4, 8, 24)),
      msm_drm.struct_drm_msm_submitqueue: (12, (0, 4, 8)),
    }

    for struct_type, (size, offsets) in layouts.items():
      with self.subTest(struct=struct_type.__name__):
        self.assertEqual(struct_type.SIZE, size)
        self.assertEqual(ctypes.sizeof(struct_type), size)
        self.assertEqual(tuple(field[2] for field in struct_type._real_fields_), offsets)

  def test_ioctl_numbers_include_linux_struct_sizes(self):
    def ioctl_number(ioctl):
      direction, base, number, struct_type = ioctl.args
      return direction << 30 | ctypes.sizeof(struct_type) << 16 | base << 8 | number

    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GET_PARAM), 0xC0186440)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT), 0xC0486446)
    self.assertEqual(ioctl_number(msm_drm.DRM_IOCTL_MSM_WAIT_FENCE), 0x40206447)


if __name__ == "__main__":
  unittest.main()
