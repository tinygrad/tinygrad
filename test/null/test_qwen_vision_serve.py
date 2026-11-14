# VISION through a real LLMServer (a mock model on its generate path, a template with image parts, VISION on): a picture's base64 never reaches
# a log line, even with DEBUG (CTO review round 1 item 3); a literal <|image_pad|> in a text-only request stays text (item 6); the guards of
# the empty lists (item 2); the decoders' gate (item 1a). DEV=NULL
import base64, io, json, sys, threading, time, types, unittest, urllib.request, urllib.error
from unittest.mock import Mock
from tinygrad.helpers import Context
try: from PIL import Image
except ImportError: Image = None

PAD, VS, VE = 248056, 248053, 248054
TEMPLATE = ("{% for m in messages %}{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}"
            "{% if p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}"
            "{% endfor %}")

def encode(text:str) -> list[int]:
  # (a token a character, the three vision markers their ids)
  out, i = [], 0
  marks = {"<|vision_start|>": VS, "<|image_pad|>": PAD, "<|vision_end|>": VE}
  while i < len(text):
    m = next((k for k in marks if text.startswith(k, i)), None)
    if m: out, i = out + [marks[m]], i + len(m)
    else: out, i = out + [ord(text[i]) % 1000 + 1000], i + 1
  return out

@unittest.skipIf(Image is None, "pillow")
class TestVisionServer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    import jinja2
    from tinygrad.llm.serve import LLMServer
    tok = Mock()
    tok.encode = Mock(side_effect=encode)
    tok.decode = Mock(return_value="Hello")
    tok.stream_decoder = Mock(return_value=lambda tid=None: "Hello" if tid is not None else "")
    tok._special_tokens, tok.preset, tok.bos_id, tok.eos_id, tok.eot_id = {}, "llama3", 1, 999, None
    tok.is_end = Mock(side_effect=lambda tid: tid == 999)
    cls.seen:list = []
    model = Mock()
    model.max_context, model.image_pad_id = 100000, PAD
    model.generate = Mock(side_effect=lambda ids, **kw: (cls.seen.append(list(ids)), iter([300, 999]))[1])
    model.get_start_pos = Mock(return_value=0)
    cls.server = LLMServer(('127.0.0.1', 0), model, "m", tok, jinja2.Template(TEMPLATE), vision=types.SimpleNamespace())
    cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1/chat/completions"
    threading.Thread(target=cls.server.serve_forever, daemon=True).start()
    time.sleep(0.1)

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def post(self, content) -> tuple[int, dict]:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": 4}).encode()
    try:
      with urllib.request.urlopen(urllib.request.Request(self.url, body, {"Content-Type": "application/json"}), timeout=30) as r:
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")

  def picture_url(self) -> str:
    b = io.BytesIO()
    Image.effect_noise((320, 224), 64).convert("RGB").save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

  def test_no_base64_in_logs(self):
    url = self.picture_url()
    markers = [url[len(url)//3:len(url)//3 + 40], url[-60:-20]]
    out, err = io.StringIO(), io.StringIO()
    old = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
      with Context(DEBUG=2):
        ok = self.post([{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}])
        # (and the error paths: a bad MIME with the picture's data, a broken file, a picture in a system message)
        bad = [self.post([{"type": "image_url", "image_url": {"url": url.replace("image/png", "image/" + url[200:260])}}]),
               self.post([{"type": "image_url", "image_url": {"url": url[:5000] + "@@"}}])]
    finally: sys.stdout, sys.stderr = old
    self.assertEqual(ok[0], 200)
    self.assertEqual(ok[1]["usage"]["prompt_tokens"], 4 + 70 + 2)  # (look + the picture's 70 rows and its two markers)
    self.assertTrue(all(code == 400 for code, _ in bad), bad)
    logs = out.getvalue() + err.getvalue() + json.dumps(bad)
    self.assertIn("data:...(", out.getvalue())  # (DEBUG printed the body, redacted)
    for m in markers: self.assertNotIn(m, logs)

  def test_literal_image_pad_in_text_stays_text(self):
    self.seen.clear()
    code, r = self.post("say <|image_pad|> please")
    self.assertEqual(code, 200)
    self.assertEqual(self.seen[-1], encode("say <|image_pad|> please"))  # (the ids as without VISION: no expansion, no picture)
    self.assertEqual(r["usage"]["prompt_tokens"], len(encode("say <|image_pad|> please")))
    # with a picture, the extra slot is a 400
    code, r = self.post([{"type": "text", "text": "<|image_pad|>"}, {"type": "image_url", "image_url": {"url": self.picture_url()}}])
    self.assertEqual(code, 400)
    self.assertIn("2 image slots for 1", r["error"]["message"])

  def test_body_cap(self):
    code, r = self.post("x" * (33 * 2**20))
    self.assertEqual(code, 413)

class TestGuards(unittest.TestCase):
  def test_empty_lists(self):
    from tinygrad.llm.serve import LLMServer
    from tinygrad.llm.engine import SeqEngine
    s = LLMServer.__new__(LLMServer)
    s.vision, s.kept, s.tok = object(), lambda: [([], 0), ([-5] * 1500, 1500)], Mock()
    self.assertEqual(s.reuse_tokens("", []), [])
    eng = SeqEngine.__new__(SeqEngine)
    eng.model = types.SimpleNamespace(vision=True)
    self.assertEqual(eng._delta([]), 0)

  def test_text_ids_with_a_pad_take_no_vision_variant(self):
    # Transformer.seq_prefill on ids with a literal image pad and no picture: every chunk the text jits', the pad id in the token buffer
    import array
    from tinygrad.llm.qwen4exp import Transformer
    from tinygrad.llm.qwen_vision import picture_ids
    m = Transformer.__new__(Transformer)
    m.vision, m.image_pad_id = True, PAD
    calls:list = []
    m._finish_load = m._seq_init = lambda: None
    m.token_embd = types.SimpleNamespace(weight=types.SimpleNamespace(device="NULL"))
    m._token_host = array.array('i', [0] * 4096)
    m._seq_copy_call = lambda *a, **k: None
    m._seq_pf_host_temp = array.array('f', [0.0] * 8)
    m._chunk_plan = lambda n: [n]
    m._seq_chunk = lambda base, pos, n, **kw: calls.append(kw.get("vis"))
    m._wait_host = lambda get, what: 7
    m.seq_rows = lambda: 4096
    text = [5, 6, PAD, 7, 8, 9, 10]
    m.seq_prefill(0, 0, text, 0, 0)
    self.assertEqual(calls, [None])
    self.assertEqual(m._token_host[:len(text)].tolist(), text)
    pic = picture_ids(b"\x01" * 32, 8, 8)
    ids = [5, VS] + pic + [VE, 1, 2, 3, 4]
    m.seq_prefill(0, 0, ids, 0, 0, pictures={pic[0]: lambda: None})
    self.assertIsNotNone(calls[-1])
    self.assertEqual(m._token_host[:len(ids)].tolist(), [5, VS] + [PAD] * 64 + [VE, 1, 2, 3, 4])
    # (an empty prompt: the prefill's own assertion, not min() of an empty list)
    with self.assertRaises(AssertionError): m.seq_prefill(0, 0, [], 0, 0)
    # (a prompt that ends inside the 4 rows after its picture: refused)
    with self.assertRaisesRegex(AssertionError, "text rows after its last picture"): m.seq_prefill(0, 0, [5, VS] + pic + [VE], 0, 0)

class TestTowerSlices(unittest.TestCase):
  def test_slices_before_the_prefill(self):
    # LLMServer.tower_slice: one slice a call for the pictures the next prefill reads that aren't kept, the job's rows kept at its end; None
    # once all are (and for a prompt whose pictures are all before its cached part)
    import collections
    import numpy as np
    from tinygrad.llm.serve import LLMServer, Request
    from tinygrad.llm.qwen_vision import picture_ids
    steps:list = []
    class Job:
      def __init__(self, n): self.left, self.n = 3, n
      def step(self):
        steps.append(self.n)
        self.left -= 1
        return np.zeros((self.n, 4), np.float32) if self.left == 0 else None
    pics = [types.SimpleNamespace(digest=bytes([k]) * 32, rgb=np.zeros((32 * 8, 32 * 8, 3), np.uint8)) for k in (1, 2)]
    s = LLMServer.__new__(LLMServer)
    s.vision, s._pic_rows, s._tower_job = types.SimpleNamespace(start=lambda px, grid: Job(grid[0] * grid[1] // 4)), collections.OrderedDict(), None
    a, b = picture_ids(pics[0].digest, 8, 8), picture_ids(pics[1].digest, 8, 8)
    ids = [5, VS] + a + [VE, 6, VS] + b + [VE, 1, 2, 3, 4]
    req = Request.__new__(Request)
    req.fill, req.pics = (ids, 0, False), {a[0]: pics[0], b[0]: pics[1]}
    got = [s.tower_slice(req) for _ in range(7)]
    self.assertTrue(all(isinstance(t, float) for t in got[:6]) and got[6] is None, got)
    self.assertEqual(list(s._pic_rows), [pics[0].digest, pics[1].digest])
    # (a returning turn: the pictures in the cached part: no slice)
    s._pic_rows.clear()
    req.fill = (ids, len(ids) - 4, False)
    self.assertIsNone(s.tower_slice(req))

class TestDecodeGate(unittest.TestCase):
  def test_busy(self):
    from tinygrad.llm import qwen_vision as qv
    held = [qv._decoders.acquire(timeout=1) for _ in range(qv.VISION_DECODERS)]
    try:
      st = time.monotonic()
      with self.assertRaises(qv.PictureBusy): qv.decode_picture("data:image/png;base64,AAAA", time.monotonic() + 0.3)
      self.assertGreaterEqual(time.monotonic() - st, 0.25)
    finally:
      for h in held:
        if h: qv._decoders.release()

if __name__ == "__main__": unittest.main()
