import struct, unittest
from tinygrad.runtime.autogen import mtfw
from tinygrad.runtime.support.mt.firmware import MTFirmwareImage, MTFW_VERSION_SIZE


def version(abi_type, minimum, maximum, abi_enum, name):
  return struct.pack('<II8s', abi_type | (minimum << 8) | (maximum << 20), abi_enum, name.encode())


def image(*versions):
  trailer = b''.join(versions)
  return bytes(64) + trailer + bytes(MTFW_VERSION_SIZE - len(trailer))


class TestMTFirmwareImage(unittest.TestCase):
  def test_windows_version_trailer(self):
    fw = MTFirmwareImage.from_bytes(image(version(1, 0, 1, 0, 'F-GEN'), version(2, 0, 0x66, 0x66, 'S-UQ'),
                                          version(2, 0, 0x68, 0x68, 'S-CDM')))
    self.assertEqual(len(fw.code), 64)
    self.assertEqual([(v.name, v.min_version, v.max_version, v.abi_enum) for v in fw.versions],
                     [('F-GEN', 0, 1, 0), ('S-UQ', 0, 0x66, 0x66), ('S-CDM', 0, 0x68, 0x68)])

  def test_missing_generation(self):
    with self.assertRaisesRegex(ValueError, 'generation 1'): MTFirmwareImage.from_bytes(image(version(2, 0, 0x66, 0x18, 'S-UQ')))

  def test_too_small(self):
    with self.assertRaisesRegex(ValueError, 'too small'): MTFirmwareImage.from_bytes(bytes(MTFW_VERSION_SIZE))


class TestMTFWABI(unittest.TestCase):
  def test_structure_sizes(self):
    self.assertEqual(mtfw.struct_MTFW_FWIF_TAG.SIZE, 265952)
    self.assertEqual(mtfw.MTFW_CCB_QUEUE.SIZE, 16432)
    self.assertEqual(mtfw.MTFW_CCB_ITEM.SIZE, 1456)
    self.assertEqual(mtfw.MTFW_CCB_RESP.SIZE, 48)
    self.assertEqual(mtfw.MTFW_SUBMISSION_PARA.SIZE, 1336)
    self.assertEqual(mtfw.MTFW_Q_SEMAPHORE.SIZE, 32)
    self.assertEqual(mtfw.MTFW_SUBMISSION_HEAD.SIZE, 88)
    self.assertEqual(mtfw.MTFW_KICK_HEAD.SIZE, 192)
    self.assertEqual(mtfw.struct__MTFW_FWIF_FEC_BOOT_CFG_.SIZE, 32)

  def test_submission_fields(self):
    item = mtfw.MTFW_CCB_ITEM(submissionCmd=mtfw.MTFW_SUBMISSION_CMD_GPU_CDM, ccbItemToken=0x1234)
    self.assertEqual(item.submissionCmd, mtfw.MTFW_SUBMISSION_CMD_GPU_CDM)
    self.assertEqual(item.ccbItemToken, 0x1234)
    para = mtfw.MTFW_SUBMISSION_PARA(submissionAddr=0x8000, submissionSize=0x100)
    self.assertEqual(para.submissionAddr, 0x8000)
    self.assertEqual(para.submissionSize, 0x100)


if __name__ == '__main__': unittest.main()
