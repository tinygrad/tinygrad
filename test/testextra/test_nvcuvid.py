# test_segment.hevc created with: ffmpeg -y -f lavfi -i color=c=gray:s=640x480:d=1 -vf "drawbox=x=200:y=150:w=240:h=180:color=red:t=fill" -pix_fmt nv12 -c:v libx265 -x265-params lossless=1 test_segment.hevc
import unittest, numpy as np, cv2
from pathlib import Path
from tinygrad.device import Device
from extra.nvcuvid import CUDAVideoDecoder

@unittest.skipUnless(Device.DEFAULT=="CUDA", "Testing CUDAVideoDecoder")
class TestCUDAVideoDecoder(unittest.TestCase):
  def test_hevc_decoder(self):
    dec = CUDAVideoDecoder(Device.default)
    with (Path(__file__).parent/"test_segment.hevc").open("rb") as f:
      for p in iter(lambda: f.read(4096), b""): dec.decode(p)
    if dec.frame_queue.empty(): self.fail("No frames decoded")
    nv12 = np.frombuffer(dec.frame_queue.get(), np.uint8).reshape((dec.luma_height+dec.chroma_height, dec.frame_width))
    bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    inside, outside, tol = bgr[240,320], bgr[100,100], 20
    self.assertLess(inside[0], tol, f"Blue too high: {inside[0]}")
    self.assertLess(inside[1], tol, f"Green too high: {inside[1]}")
    self.assertGreater(inside[2], 240, f"Red too low: {inside[2]}")
    for c in outside: self.assertLess(abs(int(c)-128), tol, f"Gray channel off: {c}")

if __name__=="__main__":
  unittest.main()
