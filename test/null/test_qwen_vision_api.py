# VISION's server side (serve.extract_pictures / expand_pictures / normalize_messages, qwen_vision.data_url_bytes / load_picture): what a
# request's image parts become, and every way a picture is refused (a 400 the client can act on). No model: DEV=NULL
import base64, io, unittest
from tinygrad.llm.serve import extract_pictures, expand_pictures, normalize_messages
from tinygrad.llm.qwen_vision import PictureError, load_picture, data_url_bytes, picture_runs
try: from PIL import Image
except ImportError: Image = None

def png(w:int, h:int, mode:str="RGB", color=(200, 30, 40), fmt:str="PNG", **kw) -> bytes:
  b = io.BytesIO()
  Image.new(mode, (w, h), color).save(b, format=fmt, **kw)
  return b.getvalue()

def url(data:bytes, mime:str="image/png") -> str: return f"data:{mime};base64,{base64.b64encode(data).decode()}"

def msgs(*parts, role:str="user") -> list[dict]: return [{"role": role, "content": [{"type": "text", "text": "look"}, *parts]}]
def part(u) -> dict: return {"type": "image_url", "image_url": {"url": u}}

@unittest.skipIf(Image is None, "pillow")
class TestExtract(unittest.TestCase):
  def test_one_picture(self):
    m = msgs(part(url(png(320, 224))))
    pics = extract_pictures(m)
    self.assertEqual(len(pics), 1)
    self.assertEqual(m[0]["content"][1], {"type": "image"})
    self.assertEqual(pics[0].rgb.shape, (224, 320, 3))
    self.assertEqual(pics[0].grid, (7, 10))
    # (the image part stays the template's, a video's is the note)
    m[0]["content"].append({"type": "video_url", "video_url": {"url": "x"}})
    normalize_messages(m, images=True)
    self.assertEqual([p["type"] for p in m[0]["content"]], ["text", "image", "text"])

  def test_string_url_and_jpeg_webp(self):
    for fmt, mime in (("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
      pics = extract_pictures(msgs({"type": "image_url", "image_url": url(png(300, 300, fmt=fmt), mime)}))
      self.assertEqual(pics[0].grid, (9, 9))

  def test_refused(self):
    cases = {
      "does not fetch": part("https://example.com/a.png"),
      "data: URL": part("file:///etc/passwd"),
      "not valid base64": part("data:image/png;base64,@@@"),
      "not supported": part(url(png(64, 64), "image/gif")),
      "could not be decoded": part(url(b"\x89PNG\r\n\x1a\nnot a png")),
      "animated": part(url(self._animated(), "image/webp")),
      "aspect ratio": part(url(png(4000, 16))),
      "must be a string": {"type": "image_url", "image_url": {"url": 7}},
    }
    for want, p in cases.items():
      with self.subTest(want), self.assertRaisesRegex(PictureError, want): extract_pictures(msgs(p))
    with self.assertRaisesRegex(PictureError, "system message"): extract_pictures(msgs(part(url(png(64, 64))), role="system"))
    with self.assertRaisesRegex(PictureError, "more than 8"): extract_pictures(msgs(*[part(url(png(64, 64)))] * 9))

  def _animated(self) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(b, format="WEBP", save_all=True, append_images=[Image.new("RGB", (64, 64), (9, 9, 9))])
    return b.getvalue()

  def test_bomb_refused_before_decoding(self):
    # (a PNG whose header says 10,000 x 10,000: refused by its size, its pixels never decoded)
    big = png(8, 8)
    ihdr = big.index(b"IHDR")
    import struct, zlib
    hdr = struct.pack(">II", 10000, 10000) + big[ihdr+12:ihdr+17]
    bomb = big[:ihdr] + b"IHDR" + hdr + struct.pack(">I", zlib.crc32(b"IHDR" + hdr)) + big[ihdr+21:]
    with self.assertRaisesRegex(PictureError, "pixels"): load_picture(bomb)

  def test_too_many_bytes(self):
    with self.assertRaisesRegex(PictureError, "larger than"): data_url_bytes("data:image/png;base64," + "A" * 30_000_000)

  def test_modes(self):
    # alpha on white, CMYK and palette to RGB, EXIF orientation applied (a 6: rotated 90 degrees)
    for mode, color in (("RGBA", (10, 20, 30, 0)), ("LA", (10, 0)), ("P", 3)):
      with self.subTest(mode): self.assertEqual(load_picture(png(64, 48, mode, color)).rgb.shape, (224, 320, 3))
    # (transparent on white, as qwen-vl-utils; opaque pixels as they are)
    self.assertEqual(load_picture(png(64, 48, "RGBA", (10, 20, 30, 0))).rgb[100, 100].tolist(), [255, 255, 255])
    self.assertEqual(load_picture(png(64, 48, "RGBA", (10, 20, 30, 255))).rgb[100, 100].tolist(), [10, 20, 30])
    cmyk = io.BytesIO()
    Image.new("CMYK", (64, 48), (0, 0, 0, 0)).save(cmyk, format="JPEG")
    self.assertEqual(load_picture(cmyk.getvalue()).rgb.shape[2], 3)
    exif = Image.Exif()
    exif[0x0112] = 6
    rot = io.BytesIO()
    Image.new("RGB", (640, 320), (5, 5, 5)).save(rot, format="JPEG", exif=exif.tobytes())
    self.assertEqual(load_picture(rot.getvalue()).rgb.shape[:2], (640, 320))

  def test_same_picture_same_ids(self):
    a, b = (extract_pictures(msgs(part(url(png(320, 224)))))[0] for _ in range(2))
    c = extract_pictures(msgs(part(url(png(320, 224, color=(1, 2, 3))))))[0]
    self.assertEqual(a.ids(), b.ids())
    self.assertNotEqual(a.ids()[0], c.ids()[0])

@unittest.skipIf(Image is None, "pillow")
class TestExpand(unittest.TestCase):
  def test_expand(self):
    p1, p2 = extract_pictures(msgs(part(url(png(320, 224))), part(url(png(256, 256)))))
    pad, vs, ve = 248056, 248053, 248054
    ids = [1, 2, vs, pad, ve, 3, vs, pad, ve, 4, 5, 6, 7]
    out, by_id = expand_pictures(ids, [p1, p2], pad)
    self.assertEqual(len(out), len(ids) - 2 + 70 + 64)
    self.assertEqual([(s, n) for s, n, _, _ in picture_runs(out)], [(3, 70), (3 + 70 + 3, 64)])
    self.assertEqual(set(by_id), {p1.ids()[0], p2.ids()[0]})
    with self.assertRaisesRegex(PictureError, "2 image slots for 1"): expand_pictures(ids, [p1], pad)

  def test_text_only_untouched(self):
    m = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    self.assertEqual(extract_pictures(m), [])
    normalize_messages(m, images=False)
    self.assertEqual(m[0]["content"], "ab")

if __name__ == "__main__": unittest.main()

@unittest.skipIf(Image is None, "pillow")
class TestUrlCache(unittest.TestCase):
  def test_same_bytes_decoded_once(self):
    # (a conversation sends its every earlier picture again each request: the same data: URL's picture comes from the cache, not decoded again)
    from unittest.mock import patch
    import tinygrad.llm.qwen_vision as qv
    u1, u2 = url(png(64, 48)), url(png(64, 48, color=(1, 2, 3)))
    with patch.object(qv, "load_picture", wraps=qv.load_picture) as lp:
      a = extract_pictures(msgs(part(u1), part(u2)))
      b = extract_pictures(msgs(part(u1), part(u2)))
    self.assertEqual(lp.call_count, 2)
    self.assertTrue(all(x is y for x, y in zip(a, b)) and a[0].digest != a[1].digest)
    self.assertFalse(a[0].rgb.flags.writeable)
    # (the cache's bytes bounded: VISION_URL_CACHE_MB)
    with patch.object(qv, "VISION_URL_CACHE_MB", 0):
      with patch.object(qv, "load_picture", wraps=qv.load_picture) as lp: extract_pictures(msgs(part(u1)))
    self.assertEqual(lp.call_count, 1)

@unittest.skipIf(Image is None, "pillow")
class TestReference(unittest.TestCase):
  def test_reference_is_the_cached_picture(self):
    # (a picture sent before, sent as its reference (sha256 of its data: URL): the same picture, its bytes not uploaded again)
    import hashlib
    from tinygrad.llm.qwen_vision import REF_PREFIX, PictureNotCached
    u = url(png(64, 48, color=(9, 8, 7)))
    ref = REF_PREFIX + hashlib.sha256(u.encode()).hexdigest()
    a = extract_pictures(msgs(part(u)))
    m = msgs(part(ref), role="tool")
    b = extract_pictures(m)
    self.assertIs(a[0], b[0])
    self.assertEqual(m[0]["content"][1], {"type": "image"})
    with self.assertRaises(PictureNotCached) as cm: extract_pictures(msgs(part(REF_PREFIX + "00" * 32)))
    self.assertIn("please retry your request", str(cm.exception))  # (pi's auto-retry matches these words: its retry sends the pictures in full)
    with self.assertRaises(PictureError): extract_pictures(msgs(part(REF_PREFIX + "xyz")))
