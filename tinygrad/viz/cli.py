#!/usr/bin/env python3
import argparse, pathlib, signal, sys, struct, json, os, itertools, heapq
os.environ["VIZ"] = "0"
if hasattr(signal, "SIGPIPE"): signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from typing import Iterator
from tinygrad.viz import serve as viz
from tinygrad.uop.ops import RewriteTrace
from tinygrad.helpers import temp, ansistrip, colored, time_to_str, ansilen, ProfilePointEvent, ProfileRangeEvent, TracingKey, unwrap, NO_COLOR
from tinygrad.helpers import DEBUG

# profile decoder used in CLI and tests
def decode_profile(data:bytes) -> dict:
  ret, off = data, 0
  def u(fmt:str) -> tuple:
    nonlocal off
    vals = struct.unpack_from(fmt, ret, off)
    off += struct.calcsize(fmt)
    return vals
  total_dur, global_peak, index_len, layout_len = u("<IQII")
  strings, dtypes, markers = json.loads(ret[off:off+index_len]).values()
  off += index_len
  layout:dict[str, dict] = {}
  # 0 means None, otherwise it's an enum value
  def option(i:int) -> int|None: return None if i == 0 else i-1
  for _ in range(layout_len):
    klen = u("<B")[0]
    k = ret[off:off+klen].decode()
    off += klen
    event_type, event_count = u("<BI")
    layout[k] = v = {"event_type":event_type, "events":[]}
    if event_type == 0:
      for _ in range(event_count):
        name, ref, key, st, dur, fmt = u("<IIIIfI")
        v["events"].append({"name":strings[name], "ref":option(ref), "key":option(key), "st":st, "dur":dur, "fmt":strings[fmt]})
    else:
      v["linear"] = u("<B")[0]
      v["peak"] = u("<Q")[0]
      for _ in range(event_count):
        if v["linear"]:
          ts, value = u("<IQ")
          v["events"].append({"event":"freq", "ts":ts, "value":value})
        else:
          alloc, ts, key = u("<BII")
          if alloc: v["events"].append({"event":"alloc", "ts":ts, "key":key, "arg": {"dtype":strings[u("<I")[0]], "sz":u("<Q")[0]}})
          else: v["events"].append({"event":"free", "ts":ts, "key":key, "arg": {"users":[(k, strings[rep], num, mode) \
              for k,rep,num,mode in [u("<IIIB") for _ in range(u("<I")[0])]]}})
  return {"dur":total_dur, "peak":global_peak, "layout":layout, "markers":markers}

def fmt_colored(s:str) -> str: return ansistrip(s) if NO_COLOR else s

def get(data:dict, key:str):
  for k,v in data.items():
    if ansistrip(k) == key: return v
  import difflib
  match = difflib.get_close_matches(key, [ansistrip(k) for k in data], n=1, cutoff=0.6)
  raise RuntimeError(f'item "{key}" not found in list'+(f", did you mean {match[0]!r}?" if match else ''))

def main(args) -> None:
  viz.load_rewrites(viz_data:=viz.VizData(viz.load_pickle(args.rewrites_path, default=RewriteTrace([], [], {}))))

  def fmt(val, to_str=str) -> str: return json.dumps(val if isinstance(val, dict) else {"value":val}) if args.jsonl else to_str(val)

  rewrites = {c["name"]:{s["name"]:s for s in c["steps"]} for c in viz_data.ctxs if c.get("steps")}
  def print_step(step:dict) -> None:
    data = viz.get_render(viz_data, step["query"])
    if isinstance(data.get("value"), Iterator):
      for m in data["value"]:
        if m.get("uop"): print(fmt(m["uop"]))
        if m.get("diff"):
          loc = pathlib.Path(m["upat"][0][0])
          print(fmt(f"Rewrite at {loc.parent.name}/{loc.name}:{m['upat'][0][1]}\n{m['upat'][1]}"))
          for line in m["diff"]: print(fmt(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None)))
    if data.get("src") is not None: print(fmt(data["src"]))

  # ** Graph rewrites printer
  if args.rewrites:
    if args.src is None: return print("Select a source with -s"+"\n"+"\n".join([f"  {fmt_colored(k)}" for k in rewrites]))
    steps = get(rewrites, args.src)
    if args.item is None:
      for k,v in steps.items(): print(" "*v["depth"]+k+(f" - {v['match_count']}" if v.get('match_count', 0) else ''))
    else: print_step(get(steps, args.item))
    return None

  events:list = viz.load_pickle(args.profile_path, default=[])
  if (profile_bytes:=viz.get_profile(viz_data, events)) is None: raise RuntimeError(f"empty profile in {args.profile_path}")
  profile = decode_profile(profile_bytes)
  profile["layout"].update([(f'{c["name"][5:]}{" SQTT" if s["name"].endswith("PKTS") else ""} {s["name"]}', s["data"]) for c in viz_data.ctxs
                            if c["name"].startswith("SQTT") for s in c["steps"] if s["name"].endswith(("PMC", "PKTS"))])
  if args.src is None: return print("Select a source with -s"+"\n  ALL\n"+"\n".join([f"  {fmt_colored(k)}" for k in profile["layout"]]))

  # ** SQTT printer
  data = None if args.src == "ALL" else get(profile["layout"], args.src)
  if "SQTT" in args.src:
    # modern terminals support 24-bit color
    def hex_colored(st:str, color:str) -> str: return f"\x1b[38;2;{int(color[1:3],16)};{int(color[3:5],16)};{int(color[5:7],16)}m{st}\x1b[0m"
    print(f"{'Clk':<12} {'Unit':<20} {'Op':<22} {'Dur':<4} {'Delay':<4} {'Info'}")
    print("-" * 100)
    pc_map:dict[int, str] = {}
    pkt_idxs:dict[str, itertools.count] = {}
    dispatch_to_inst:dict[str, tuple[str, int]] = {}
    inst_st:int|None = None
    for e in viz.sqtt_timeline(*data):
      if isinstance(e, ProfilePointEvent) and e.key == 'pcMap': pc_map = e.arg
      if not isinstance(e, ProfileRangeEvent): continue
      if inst_st is None: inst_st = int(e.st)
      assert isinstance(e.name, TracingKey)
      op_name, info = e.name.display_name, e.name.ret or ""
      color = next((v for k,v in viz.wave_colors.items() if k in op_name), None)
      op_str = hex_colored(op_name, color) if color and not NO_COLOR else op_name
      phase, delay = None, 0
      idx = next(pkt_idxs.setdefault(e.device, itertools.count()))
      if e.device.startswith("WAVE"):
        inst = f"0x{(pc:=int(info.replace('PC:', ''))):05x} {pc_map[pc]}" if info else f"{'':7} {op_name}"
        dispatch_to_inst[f"{e.device}-{idx}"] = (inst, int(e.st))
        phase = "DISPATCH"
      if info.startswith("LINK:"):
        inst, dispatch_st = dispatch_to_inst[info.replace("LINK:", "")]
        phase, delay = "EXEC", int(e.st) - dispatch_st
      if inst and phase: info = f"{phase:<8} {inst}"
      unit = e.device.replace(" ", "-")
      row = {"clk":int(e.st)-inst_st, "unit":unit, "op":op_name, "dur":int(unwrap(e.en)-e.st), "delay":delay or "", "info":info}
      print(fmt(row, lambda _: f"{row['clk']:<12} {unit:<20} {op_str}{' '*(22-ansilen(op_str))} {row['dur']:<4} {str(row['delay']):<4} {info}"))

  # ** PMC printer
  elif "PMC" in args.src:
    pmc = viz.unpack_pmc(data)
    cols = pmc["cols"]
    rows:list = []
    for r in pmc["rows"]:
      if args.item is None: rows.append(r[:2])
      elif args.item == r[0]:
        rows = r[2]["rows"] if len(r) > 2 else [r[:2]]
        cols = r[2]["cols"] if len(r) > 2 else cols
    pmc_data = [[x for x in cols], *[[str(x) for x in r] for r in rows]]
    widths = [max(len(r[i]) for r in pmc_data) for i in range(len(cols))]
    def pad(r): return "| "+" | ".join(x+" "*(w-len(x)) for x,w in zip(r, widths))+" |"
    table_str = pad(pmc_data[0])+"\n"+pad(["-"*w for w in widths])+"\n"+("\n".join([pad(row) for row in pmc_data[1:]]))
    print(fmt({"cols":cols, "rows":rows}, lambda _: table_str))

  # ** Memory printer
  elif data is not None and data["event_type"] == 1:
    print(fmt({"peak":data["peak"], "cols":["ts", "event", "key", "info"]},
              lambda _: f"Peak: {data['peak']}"+"\n"+f"{'TS':<10}  {'Event':<6}  {'Key':>8}  Info"))
    for e in data["events"]:
      info = str(arg:=e.pop("arg", {}))
      if e["event"] == "free":
        info = ', '.join([f"{fmt_colored(kernel)} {['read','write','write+read'][mode]}@data{num}" for _,kernel,num,mode in arg["users"]])
      print(fmt({**e, "info":info}, lambda _: f"{e['ts']:<10}  {e['event']:<6}  {e.get('key', ''):>8}  {info}"))

  # ** Profiler printer
  else:
    timelines = [(n,l) for n,l in profile["layout"].items() if l.get("event_type") == 0]
    def produce_top_kernels() -> Iterator[dict]:
      tagged = ((n,e) for n,l in timelines for e in l["events"]) if args.src == "ALL" else ((args.src,e) for e in data["events"])
      agg:dict[tuple[str,str], tuple[float, int, int|None]] = {} # map (device, kernel name) to (total time, count and ref)
      total = 0
      for dev,e in tagged:
        et = e["dur"] * 1e-3
        t, c, ref = agg.get((dev,e["name"]), (0.0, 0, None))
        agg[(dev,e["name"])] = (t+et, c+1, e["ref"])
        total += et
      items = sorted(agg.items(), key=lambda kv:kv[1][0], reverse=True)
      num_rows = len(items) if args.top < 0 else args.top
      for (dev,name),(t,c,ref) in items[:num_rows]:
        display = f"{dev[:7]:7s} {fmt_colored(name)}" if args.src == "ALL" else fmt_colored(name)
        yield {"name":display, "dur_ms":t, "count":c, "pct":t/total*100.0, "ref":ref}
      if num_rows > 0 and items[num_rows:]:
        other_t = sum(t for _,(t,_,_) in items[num_rows:])
        other_c = sum(c for _,(_,c,_) in items[num_rows:])
        yield {"name":"Other", "dur_ms":other_t, "count":other_c, "pct":other_t/total*100.0, "ref":None}
    def produce_all_kernels() -> Iterator[dict]:
      event_streams = [[(e["st"], n, e) for e in l["events"]] for n,l in timelines] if args.src == "ALL" \
                      else [[(e["st"], args.src, e) for e in data["events"]]]
      marker_stream = sorted([(m["ts"], "MARKER", m) for m in profile.get("markers", [])], key=lambda t:t[0])
      for ts,dev,e in heapq.merge(*event_streams, marker_stream, key=lambda t:t[0]):
        if dev == "MARKER":
          yield {"device":dev, "name":fmt_colored(e["name"]), "et_ms":ts*1e-3, "ref":None, "ext":None}
          continue
        ext:list[str] = []
        if (fmt:=e["fmt"]).startswith("TB:"):
          tb, fmt = json.loads(e["fmt"].replace("TB:", "")), ""
          while tb:
            file, lineno, fxn, code = tb.pop()
            line = f"{file.split('/')[-1]}:{lineno} {fxn}"
            if fmt: ext.append(f"{line} {code}")
            elif not file.startswith("<") and not fxn.startswith("<"): fmt = line
        yield {"device":dev, "name":fmt_colored(e["name"]), "dur_ms":e["dur"]*1e-3,
               "et_ms":(e["st"]+e["dur"])*1e-3, "fmt":fmt, "ref":e["ref"], "ext":"\n".join(ext)}
    def fmt_top(k:dict) -> str:
      return f"{fmt_colored(k['name'])}{' ' * max(0, 36-ansilen(k['name']))} {time_to_str(k['dur_ms']*1e-3, w=9)} {k['count']:7d} {k['pct']:6.2f}%"
    def fmt_all(k:dict) -> str:
      if k["device"] == "MARKER": return f"--- MARKER {k['name']} /{k['et_ms']:9.2f}ms"
      ptm = colored(time_to_str(k["dur_ms"]*1e-3, w=9), "yellow" if k["dur_ms"] > 10 else None)
      fmt_str = "  ".join(p+" "*max(0, 14-ansilen(p)) for p in k["fmt"].split("\n"))
      name = f"*** {k['device'][:7]:7s} "+k["name"]+" "*(46-ansilen(k["name"]))
      return f"{name} tm {ptm}/{k['et_ms']:9.2f}ms"+(f" ({fmt_str})" if k["fmt"] else "")
    fmt_row = fmt_top if args.top else fmt_all
    seen_refs:set[int] = set()
    for k in (produce_top_kernels if args.top else produce_all_kernels)():
      print(fmt(k, to_str=fmt_row))
      if k["ref"] is not None and k["ref"] not in seen_refs:
        seen_refs.add(k["ref"])
        steps = rewrites[viz_data.ctxs[k["ref"]]["name"]]
        if DEBUG >= 3 and (ast_step:=steps.get("View Base AST")) is not None: print_step(ast_step)
        if DEBUG >= 4 and (src_step:=steps.get("View Source")) is not None: print_step(src_step)
      elif DEBUG >= 3 and k.get("ext"): print(fmt(k["ext"]))

def get_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(add_help=False)
  g_mode = parser.add_argument_group("mode")
  g_mode.add_argument("-p", "--profile", action="store_true", help="View profile")
  g_mode.add_argument("-r", "--rewrites", action="store_true", help="View graph rewrites")
  g_opts = parser.add_argument_group("optional args")
  g_opts.add_argument("-s", "--src", type=str, default=None, metavar="NAME", help="Select a data source (default: list all sources)")
  g_opts.add_argument("-i", "--item", type=str, default=None, metavar="NAME", help="Select an item within the source (default: list all items)")
  g_opts.add_argument("-t", "--top", type=int, default=None, metavar="COUNT",
                      help="Number of top kernels to aggregate (default: do not aggregate, set -1 to aggregate all)")
  g_opts.add_argument("--profile-path", type=pathlib.Path, metavar="PATH", help="Path to profile.pkl (optional file, default: latest profile)",
                      default=pathlib.Path(temp("profile.pkl", append_user=True)))
  g_opts.add_argument("--rewrites-path", type=pathlib.Path, metavar="PATH", help="Path to rewrites.pkl (optional file, default: latest rewrites)",
                      default=pathlib.Path(temp("rewrites.pkl", append_user=True)))
  g_opts.add_argument("--jsonl", action="store_true", help="Emit profiler output as JSONL")
  g_opts.add_argument("-h", "--help", action="help", help="show this help message and exit")
  return parser

if __name__ == "__main__":
  args = get_arg_parser().parse_args()
  if not args.profile and not args.rewrites:
    get_arg_parser().print_help()
    sys.exit(0)

  try: main(args)
  except KeyboardInterrupt: pass
