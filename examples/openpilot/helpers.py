import io, pickle, shutil, struct, tempfile, codecs, json, urllib.request
from typing import Any
from tinygrad.helpers import fetch
from tinygrad.nn.onnx import OnnxPBParser

MODELS = {
  'driving': ('65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce', 60918562),
  'dm': ('dd299afabe7a3e0d04cbe2bd97fdb0c93bba8ad6d3cc3663a0e0ededaf243ac2', 7497335),
  'big': ('6fee5937923c74848df4a63f6239eb6331c6274dd4bdb7a5d6ec0388a8b543d5', 766018462),
}

def fetch_model(name):
  sha, size = MODELS[name]
  request = urllib.request.Request('https://huggingface.co/commaai/openpilot-lfs.git/info/lfs/objects/batch',
    data=json.dumps({'operation': 'download', 'transfers': ['basic'], 'objects': [{'oid': sha, 'size': size}]}).encode(),
    headers={'Content-Type': 'application/vnd.git-lfs+json'})
  with urllib.request.urlopen(request) as response: url = json.load(response)['objects'][0]['actions']['download']['href']
  return fetch(url, name=f'openpilot_{sha}', sha256=sha)

def dump_oob(obj, f):
  with tempfile.TemporaryFile(dir=".") as tmp:
    def buffer_callback(pb: pickle.PickleBuffer):
      m = pb.raw()
      tmp.write(struct.pack('<q', m.nbytes))
      tmp.write(m)
      pb.release() # keep peak ram at ~1 buffer
    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)

def load_oob(f):
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    while (h := f.read(8)):
      pb = pickle.PickleBuffer(bytearray(struct.unpack('<q', h)[0]))
      if f.readinto(pb) != pb.raw().nbytes:
        raise EOFError("incomplete model buffer")
      yield pb
  return pickle.load(io.BytesIO(opcodes), buffers=buffers())


class MetadataOnnxPBParser(OnnxPBParser):
  def _parse_ModelProto(self) -> dict:
    obj: dict[str, Any] = {"graph": {"input": [], "output": []}, "metadata_props": []}
    for fid, wire_type in self._parse_message(self.reader.len):
      match fid:
        case 7:
          obj["graph"] = self._parse_GraphProto()
        case 14:
          obj["metadata_props"].append(self._parse_StringStringEntryProto())
        case _:
          self.reader.skip_field(wire_type)
    return obj


def get_name_and_shape(value_info: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
  shape = tuple(int(dim) if isinstance(dim, int) else 0 for dim in value_info["parsed_type"].shape)
  name = value_info["name"]
  return name, shape


def get_metadata_value_by_name(model: dict[str, Any], name: str) -> str | Any:
  for prop in model["metadata_props"]:
    if prop["key"] == name:
      return prop["value"]
  return None


def make_metadata_dict(model_path):
  model = MetadataOnnxPBParser(model_path).parse()
  output_slices = get_metadata_value_by_name(model, 'output_slices')
  return {
    'model_checkpoint': get_metadata_value_by_name(model, 'model_checkpoint'),
    'output_slices': pickle.loads(codecs.decode(output_slices.encode(), "base64")) if output_slices is not None else None,
    'input_shapes': dict(get_name_and_shape(x) for x in model["graph"]["input"]),
    'output_shapes': dict(get_name_and_shape(x) for x in model["graph"]["output"]),
  }


if __name__ == '__main__':
  import sys
  print(fetch_model(sys.argv[1]))
