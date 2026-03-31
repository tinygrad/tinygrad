#!/usr/bin/env python3
import argparse, pathlib, signal, sys, struct, json, itertools
if hasattr(signal, "SIGPIPE"): signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from typing import Iterator
from tinygrad.viz import serve as viz
from tinygrad.uop.ops import RewriteTrace
from tinygrad.helpers import temp, ansistrip, colored, time_to_str, ansilen, ProfilePointEvent, ProfileRangeEvent, TracingKey, unwrap

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
    v:dict = {"events":[]}
    layout[k] = v
    event_type, event_count = u("<BI")
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
          else: v["events"].append({"event":"free", "ts":ts, "key":key, "arg": {"users":[u("<IIIB") for _ in range(u("<I")[0])]}})
  return {"dur":total_dur, "peak":global_peak, "layout":layout, "markers":markers}

def get(data:dict, key:str):
  for k,v in data.items():
    if ansistrip(k) == key: return v
  raise RuntimeError(f'item "{key}" not found in list')

def main(args) -> None:
  viz.trace = viz.load_pickle(args.rewrites_path, default=RewriteTrace([], [], {}))
  viz.ctxs = viz.get_rewrites(viz.trace)

  def format_colored(s:str) -> str: return ansistrip(s) if args.no_color else s

  if args.profile:
    events:list = viz.load_pickle(args.profile_path, default=[])
    if (profile_bytes:=viz.get_profile(events)) is None: raise RuntimeError(f"empty profile in {args.profile_path}")
    profile = decode_profile(profile_bytes)
    viz.load_amd_counters(viz.ctxs, events)
    profile["layout"].update([(f'{c["name"]} {s["name"]}', s["data"]) for c in viz.ctxs if c["name"].startswith("SQTT") for s in c["steps"]
                              if s["name"].startswith("PKTS")])
    if args.src is None:
      for k in profile["layout"]:
        print(f"  {format_colored(k)}")
      return None

    # ** SQTT printer
    data = get(profile["layout"], args.src)
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
        op_str = hex_colored(op_name, color) if color and not args.no_color else op_name
        phase, delay = None, 0
        idx = next(pkt_idxs.setdefault(e.device, itertools.count()))
        if e.device.startswith("WAVE") or e.device == "OTHER":
          inst = f"0x{(pc:=int(info.replace('PC:', ''))):05x} {pc_map[pc]}" if info else f"{'':7} {op_name}"
          dispatch_to_inst[f"{e.device}-{idx}"] = (inst, int(e.st))
          phase = "DISPATCH"
        if info.startswith("LINK:"):
          inst, dispatch_st = dispatch_to_inst[info.replace("LINK:", "")]
          phase, delay = "EXEC", int(e.st) - dispatch_st
        if inst and phase: info = f"{phase:<8} {inst}"
        unit = e.device.replace(" ", "-")
        print(f"{int(e.st)-inst_st:<12} {unit:<20} {op_str}{' '*(22-ansilen(op_str))} {int(unwrap(e.en)-e.st):<4} {str(delay or ''):<4} {info}")
      return None

    # ** Profiler printer
    agg:dict[str, tuple[float, int]] = {}
    total = 0
    for e in data.get("events", []):
      et = e["dur"] * 1e-6
      if args.item is not None:
        if ansistrip(e["name"]) == args.item:
          ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None)
          name = e["name"] + (" " * (46 - ansilen(e["name"])))
          print(f"{name} {ptm}/{et*1e3:9.2f}ms  " + e.get("fmt", "").replace("\n", " | ") + "  ")
      else:
        t, c = agg.get(e["name"], (0.0, 0))
        agg[e["name"]] = (t+et, c+1)
        total += et
    if agg and total > 0:
      from tabulate import tabulate
      items = sorted(agg.items(), key=lambda kv:kv[1][0], reverse=True)
      table = [[name, time_to_str(t, w=9), c, f"{(t/total*100.0):.2f}%"] for name,(t,c) in items]
      print(tabulate(table, headers=["name", "total", "count", "pct"], tablefmt="github"))
    return None

  # ** Graph rewrites printer
  rewrites = {c["name"]:{s["name"]:s for s in c["steps"]} for c in viz.ctxs if c.get("steps")}
  if args.src is None:
    for k in rewrites: print(f"  {format_colored(k)}")
    return None
  steps = get(rewrites, args.src)
  if args.item is None:
    for k,v in steps.items(): print(" "*v["depth"]+k+(f" - {v['match_count']}" if v.get('match_count', 0) else ''))
  else:
    data = viz.get_render(get(steps, args.item)["query"])
    if isinstance(data.get("value"), Iterator):
      for m in data["value"]:
        if m.get("uop"): print(f"Input UOp:\n{m['uop']}")
        if m.get("diff"):
          loc = pathlib.Path(m["upat"][0][0])
          print(f"Rewrite at {loc.parent.name}/{loc.name}:{m['upat'][0][1]}\n{m['upat'][1]}")
          for line in m["diff"]: print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
    if data.get("src") is not None: print(data["src"])

def get_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  g_mode = parser.add_argument_group("mode")
  g_mode.add_argument("--profile", action="store_true", help="View profile")
  g_mode.add_argument("--rewrites", action="store_true", help="View graph rewrites")
  g_opts = parser.add_argument_group("options")
  g_opts.add_argument("-s", "--src", type=str, default=None, metavar="NAME", help="Select a data source (default: list all sources)")
  g_opts.add_argument("-i", "--item", type=str, default=None, metavar="NAME", help="Select an item within the source (default: list all items)")
  g_opts.add_argument("--no-color", action="store_true", help="Turn off colored names")
  g_opts.add_argument("--profile-path", type=pathlib.Path, metavar="PATH", help="Path to profile.pkl (optional file, default: latest profile)",
                      default=pathlib.Path(temp("profile.pkl", append_user=True)))
  g_opts.add_argument("--rewrites-path", type=pathlib.Path, metavar="PATH", help="Path to rewrites.pkl (optional file, default: latest rewrites)",
                      default=pathlib.Path(temp("rewrites.pkl", append_user=True)))
  return parser

if __name__ == "__main__":
  args = get_arg_parser().parse_args()
  if not args.profile and not args.rewrites:
    get_arg_parser().print_help()
    sys.exit(0)

  try: main(args)
  except KeyboardInterrupt: pass
