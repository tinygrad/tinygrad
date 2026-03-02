#!/usr/bin/env python3
import os
os.environ["VIZ"] = "0"
import argparse, pathlib, sys, struct, json
from typing import Iterator
from tinygrad.viz import serve as viz
from tinygrad.uop.ops import RewriteTrace
from tinygrad.helpers import temp, ansistrip, colored, time_to_str, ansilen

# ** generic helpers

def optional_eq(val:dict, arg:str|None) -> bool: return arg is None or ansistrip(val["name"]) == arg

def print_data(data:dict) -> None:
  if isinstance(data.get("value"), Iterator):
    for m in data["value"]:
      if m.get("uop"): print(f"Input UOp:\n{m['uop']}")
      if m.get("diff"):
        loc = pathlib.Path(m["upat"][0][0])
        print(f"Rewrite at {loc.parent.name}/{loc.name}:{m['upat'][0][1]}\n{m['upat'][1]}")
        for line in m["diff"]: print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
  if data.get("src") is not None: print(data["src"])

# ** Profiler trace decoder

# 0 means None, otherwise it's an enum value
def option(i:int) -> int|None: return None if i == 0 else i-1

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
  for _ in range(layout_len):
    klen = u("<B")[0]
    k = ret[off:off+klen].decode()
    off += klen
    layout[k] = v = {"events":[]}
    event_type, event_count = u("<BI")
    if event_type == 0:
      for _ in range(event_count):
        name, ref, key, st, dur, fmt = u("<IIIIfI")
        v["events"].append({"name":strings[name], "ref":option(ref), "key":option(key), "st":st, "dur":dur, "fmt":strings[fmt]})
    else:
      v["peak"] = u("<Q")[0]
      for _ in range(event_count):
        alloc, ts, key = u("<BII")
        if alloc: v["events"].append({"event":"alloc", "ts":ts, "key":key, "arg": {"dtype":strings[u("<I")[0]], "sz":u("<Q")[0]}})
        else: v["events"].append({"event":"free", "ts":ts, "key":key, "arg": {"users":[u("<IIIB") for _ in range(u("<I")[0])]}})
  return {"dur":total_dur, "peak":global_peak, "layout":layout, "markers":markers}

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  g_mode = parser.add_argument_group("mode")
  g_mode.add_argument("--profile", action="store_true", help="View profile trace")
  g_mode.add_argument("--rewrites", action="store_true", help="View rewrites trace")
  g_common = parser.add_argument_group("common options")
  g_common.add_argument("--kernel", type=str, default=None, metavar="NAME", help="Select a kernel by name (optional name, default: only list names)")
  g_profile = parser.add_argument_group("profile options")
  g_profile.add_argument("--device", type=str, default=None, metavar="NAME", help="Select a device (optional name, default: only list names)")
  g_profile.add_argument("--top", type=int, default=10, metavar="N", help="Number of top kernels to show (-1 for all, default: 10)")
  g_rewrites = parser.add_argument_group("rewrites options")
  g_rewrites.add_argument("--select", type=str, default=None, metavar="NAME",
                          help="Select an item within the chosen kernel (optional name, default: only list names)")
  parser.add_argument("--profile-path", type=pathlib.Path, metavar="PATH", help="Path to profile (optional file, default: latest profile)",
                        default=pathlib.Path(temp("profile.pkl", append_user=True)))
  parser.add_argument("--rewrites-path", type=pathlib.Path, metavar="PATH", help="Path to rewrites (optional file, default: latest rewrites)",
                        default=pathlib.Path(temp("rewrites.pkl", append_user=True)))
  args = parser.parse_args()
  if not args.profile and not args.rewrites:
    parser.print_help()
    sys.exit(0)

  viz.trace = viz.load_pickle(args.rewrites_path, default=RewriteTrace([], [], {}))
  viz.ctxs = viz.get_rewrites(viz.trace)

  if args.profile:
    from tabulate import tabulate
    profile = decode_profile(viz.get_profile(viz.load_pickle(args.profile_path, default=[])))
    agg, total, n = {}, 0, 0
    if args.device is None: print("Select a device:")
    for k,v in profile["layout"].items():
      if not optional_eq({"name":k}, args.device): continue
      print(f"  {k}")
      if args.device is None: continue
      for e in v.get("events", []):
        et = e["dur"]*1e-6
        if args.kernel is not None:
          if optional_eq(e, args.kernel) and n < 10:
            ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
            name = e["name"]+(" " * (46 - ansilen(e["name"])))
            print(f"{name} {ptm}/{(et or 0)*1e3:9.2f}ms  "+e['fmt'].replace('\n', ' | ')+"  ")
            n += 1
        else:
          a = agg.setdefault(e["name"], [0.0, 0])
          a[0] += et
          a[1] += 1
          total += et
    if agg and total > 0:
      items = sorted(agg.items(), key=lambda kv:kv[1][0], reverse=True)
      sel = items if args.top == -1 else items[:args.top]
      table = [[name, time_to_str(t, w=9), c, f"{(t/total*100.0):.2f}%"] for name,(t,c) in sel]
      if args.top != -1 and (other:=items[len(sel):]):
        other_t = total-sum(t for _, (t, _) in sel)
        table.append([f"Other ({len(other)} unique)", time_to_str(other_t, w=9), sum(c for _,(_,c) in other), f"{other_t/total*100.0:.2f}%"])
      print(tabulate(table, headers=["name", "total", "count", "pct"], tablefmt="github"))
    sys.exit(0)

  for k in viz.ctxs:
    if not optional_eq(k, args.kernel): continue
    print(k["name"])
    if args.kernel is None: continue
    for s in k["steps"]:
      if not optional_eq(s, args.select): continue
      print(" "*s["depth"]+s['name']+(f" - {s['match_count']}" if s.get('match_count') is not None else ''))
      if args.select is not None: print_data(viz.get_render(s['query']))
