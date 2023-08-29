import os, sys
from collections import defaultdict
from typing import Set
from tinygrad.helpers import TINYBOARD, getenv
import requests, time, threading, json, inspect, secrets # type: ignore
import networkx as nx

TINYBOARD_HOST = getenv("TINYBOARD_HOST", "http://127.0.0.1:6226")
class FuncVisualContext:
  def __init__(self, func):
    self.func = func
    self.mlops_graph = nx.DiGraph()
    self.lazyops_graph = nx.DiGraph()
    self.lazyops_clusters = []
    self.mlops_info = []

class TinyBoardConnector:
  def __init__(self):
    self.sess_id = secrets.token_urlsafe(16)
    self.payload_queue = []
    self.graphs = defaultdict(list)
    self.debug_context = []
    self.visited_funcs = set()
    self.obj_enumerator = 0

    env_res = ""
    for k,v in os.environ.items(): env_res += f"{k}={v}"
    self.log("init", {"session": self.sess_id, "args": ' '.join(sys.argv), "name": getenv("TINYBOARD_NAME", ' '.join(sys.argv))})

  def log(self, name, payload):
    self.payload_queue.append({"name": name, "payload": payload})

  @staticmethod
  def worker(conn):
    if len(conn.graphs) > 0:
      graphs = {}
      # Merge graphs for efficiency
      for k,vv in conn.graphs.items():
        for v in vv:
          nv = {"name": v[0], "type": v[1], "series": v[2], "xaxis": v[3] if v[3] is not None else [], "graphinfo": v[4]}
          if k not in graphs:
            graphs[k] = nv
          else:
            for i in range(len(graphs[k]['series'])):
              graphs[k]['series'][i] += nv['series'][i]
            for i in range(len(graphs[k]['xaxis'])):
              graphs[k]['xaxis'][i] += nv['xaxis'][i]
      for k,v in graphs.items():
        nv = {"name": v['name'], "type": v['type'], "series": json.dumps(v['series']), "xaxis": json.dumps(v['xaxis']), "graphinfo": json.dumps(v['graphinfo'])}
        conn.log("graph", nv)
      conn.graphs = defaultdict(list)

    if len(conn.payload_queue) > 0:
      url = TINYBOARD_HOST + "/api/log"
      requests.post(url, json={"session": conn.sess_id, "payloads": conn.payload_queue})
      conn.payload_queue = []

the_conn = None
def __get_conn():
  global the_conn
  if the_conn is None: the_conn = TinyBoardConnector()
  return the_conn

def nm(x):
  global the_conn
  assert the_conn is not None
  if not hasattr(x, 'node_id'):
    setattr(x, 'node_id', the_conn.obj_enumerator)
    the_conn.obj_enumerator += 1
  return x.node_id

def __is_lazybuffer(x):
  return hasattr(x, 'output_buffer') # Check if x is a LB without any imports.

def __travel_lazyop(gx, nodes, op):
  if __is_lazybuffer(op): return
  if nm(op) not in gx.nodes: gx.add_node(nm(op))
  nodes.add(nm(op))
  gx.nodes[nm(op)]['color'] = 'black'
  gx.nodes[nm(op)]['label'] = str(op.op)
  for x in op.src:
    if __is_lazybuffer(x): gx.add_edge(nm(x.op), nm(op), label="", color='#000000')
    else: gx.add_edge(nm(x), nm(op), label="", color='#000000')
    __travel_lazyop(gx,nodes, x)

def tinyboard_log(name, payload):
  if not TINYBOARD: return
  __get_conn().log(name, payload)

def tinyboard_log_kernel(name, prg):
  if not TINYBOARD: return
  tinyboard_log("kernel_def", {"name":name, "src":prg})

def tinyboard_log_mlops(fxn, ret, *xs):
  if not TINYBOARD: return
  the_conn = __get_conn()
  for gx in the_conn.debug_context:
    # Building mlops graph.
    for x in xs: gx.mlops_graph.add_edge(nm(x), nm(ret), color='#000000')
    if nm(ret) not in gx.mlops_graph.nodes:
      gx.mlops_graph.add_node(nm(ret))
    ret_label = str(fxn.__name__).replace('"', "'") + f"({nm(ret)})"
    gx.mlops_graph.nodes[nm(ret)]['label'] = f'"{ret_label}"'
    gx.mlops_graph.nodes[nm(ret)]['color'] = '#348090'

    # Collecting all lazyops inside mlop.
    nodes: Set[int] = set()
    __travel_lazyop(gx.lazyops_graph, nodes, ret.lazydata.op)

    # Subcluster lazyops from the same mlops to visualize it.
    gx.lazyops_clusters.append(f"""subgraph cluster_{nm(ret)} {{
      style=filled;
      color="#e0e0e0";
      {' '.join(map(str, list(nodes)))};
      label="{str(fxn)}";
    }}""")

    # Collecting frames for inspected function. This collects stack for every mlop.
    stack, tensor_function, tensor_frame_uid = [], "", 0
    for (frame,filename,line_number,function_name,lines,index) in inspect.stack()[2:]:
      if frame.f_locals.get('function', None)==gx.func: break
      if filename.endswith("tinygrad/tensor.py"):
        # tensor_frame_uid a key to make frame call to the same functions unique.
        tensor_frame_uid = str(abs((id(frame),function_name,frame.f_back.f_lasti, frame.f_back.f_lineno).__hash__())) # type: ignore
        tensor_function = function_name
      stack.append([function_name, filename, line_number, ''.join(l.strip().rstrip() for l in lines) if lines else ''])
    gx.mlops_info.append({"node": nm(ret), "tensor_function": tensor_function, "tensor_frame_uid": tensor_frame_uid, "stack": stack})

def tinyboard_log_graph(name, type, series, xaxis=None, graphinfo=None):
  if not TINYBOARD: return
  assert len(series) > 0, "Series could not be empty"
  __get_conn().graphs[name].append((name, type, series, xaxis, graphinfo))

def tinyboard_inspector():
  def decorator(function):
    def wrapper(*args, **kwargs):
      if not TINYBOARD: return function(*args, **kwargs)
      the_conn = __get_conn()
      if function in the_conn.visited_funcs: return function(*args, **kwargs)
      the_conn.debug_context.append(FuncVisualContext(function))
      result = function(*args, **kwargs)
      mlops_graph = nx.drawing.nx_pydot.to_pydot(the_conn.debug_context[-1].mlops_graph).to_string()
      mlops_info = json.dumps(the_conn.debug_context[-1].mlops_info)
      lazyops_graph = nx.drawing.nx_pydot.to_pydot(the_conn.debug_context[-1].lazyops_graph).to_string()[:-2]
      lazyops_graph += '\n'.join(the_conn.debug_context[-1].lazyops_clusters) + "}"
      the_conn.log(f"func_debug", {"function_name": function.__name__, "mlops_graph": mlops_graph, "lazyops_graph": lazyops_graph, "mlops_info": mlops_info})
      the_conn.debug_context.pop()
      the_conn.visited_funcs.add(function)
      return result
    return wrapper
  return decorator

def __sync_tinyboard():
  while True:
    for _ in range(10):
      if not threading.main_thread().is_alive(): break
      time.sleep(0.5)
    if the_conn:
      TinyBoardConnector.worker(the_conn)
      if not threading.main_thread().is_alive(): return
if TINYBOARD: threading.Timer(5.0, __sync_tinyboard).start()