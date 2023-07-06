from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.versions_pb2 import VersionDef

from tinygrad.graph import nm, get_sop
from tinygrad.lazy import LazyBuffer
from tinygrad.ops import LazyOp

def proto_attr(shape, s=None):
  attr = {}
  if s is not None: attr["attr"] = AttrValue(s=s.encode(encoding="utf_8"))
  if shape is not None: attr["_output_shapes"] = AttrValue(list=AttrValue.ListValue(shape=[TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in shape])]))
  return attr
def inp_name(x): return f"input/{nm(x)}"
def op_to_graph(_ret: LazyBuffer, _ast: LazyOp):
  nodes = []
  def add_node(ret: LazyBuffer, ast: LazyOp):
    for x in ast.buffers:
      if isinstance(x.realized, LazyOp): add_node(x, x.realized)
      else: nodes.append(NodeDef(name=inp_name(x), op="IO Node", input=[], attr=proto_attr(x.shape)))
    nodes.append(NodeDef(name=str(nm(ret)), op=get_sop([l.op for l in ast.get_lazyops()]), input=[inp_name(x) for x in ast.buffers], attr=proto_attr(ret.shape)))
  add_node(_ret, _ast)
  return GraphDef(node=nodes, versions=VersionDef(producer=22)), RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device=f"/device:{_ret.device}:0")]))