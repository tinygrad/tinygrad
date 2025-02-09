# test_segment.hevc created with: ffmpeg -y -f lavfi -i color=c=gray:s=640x480:d=1 -vf "drawbox=x=200:y=150:w=240:h=180:color=red:t=fill" -pix_fmt nv12 -c:v libx265 -x265-params lossless=1 test_segment.hevc
import unittest, numpy as np
from pathlib import Path
from tinygrad.device import Device
from extra.nvcuvid import CUDAVideoDecoder

@unittest.skipUnless(Device.DEFAULT=="CUDA", "CUDA required")
class TestCUDAVideoDecoder(unittest.TestCase):
  def test_hevc_decoder(self):
    dec = CUDAVideoDecoder(Device.default)
    with (Path(__file__).parent/"test_segment.hevc").open("rb") as f:
      for p in iter(lambda: f.read(4096), b""): dec.decode(p)
    if dec.frame_queue.empty(): self.fail("No frames")
    frame = np.frombuffer(dec.frame_queue.get(), np.uint8)\
              .reshape((dec.luma_height+dec.chroma_height, dec.frame_width))
    Y, UV = frame[:dec.luma_height], frame[dec.luma_height:]
    tol = 10
    for y in range(0, dec.luma_height, 50):
      for x in range(0, dec.frame_width, 50):
        if 150 <= y < 330 and 200 <= x < 440: continue
        uv_row, uv_col = y//2, (x//2)*2
        self.assertLess(abs(int(Y[y,x])-128), tol, f"Gray Y off at ({y},{x})")
        self.assertLess(abs(int(UV[uv_row,uv_col])-128), tol, f"Gray U off at ({y},{x})")
        self.assertLess(abs(int(UV[uv_row,uv_col+1])-128), tol, f"Gray V off at ({y},{x})")
    for y in range(150, 330, 20):
      for x in range(200, 440, 20):
        uv_row, uv_col = y//2, (x//2)*2
        self.assertLess(int(Y[y,x]), 100, f"Red Y too high at ({y},{x})")
        self.assertLess(int(UV[uv_row,uv_col]), 100, f"Red U too high at ({y},{x})")
        self.assertGreaterEqual(int(UV[uv_row,uv_col+1]), 240, f"Red V too low at ({y},{x})")

if __name__=="__main__":
  unittest.main()
