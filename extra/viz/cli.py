#!/usr/bin/env python3
import argparse, pathlib, sys, struct, json, itertools
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  g_mode = parser.add_argument_group("mode")
  g_mode.add_argument("--profile", action="store_true", help="View profile trace")
  g_mode.add_argument("--rewrites", action="store_true", help="View rewrites trace")
  g_common = parser.add_argument_group("common options")
  g_common.add_argument("--kernel", type=str, default=None, metavar="NAME", help="Select a kernel by name (optional name, default: only list names)")
  g_profile = parser.add_argument_group("profile options")
  g_profile.add_argument("--device", type=str, default=None, metavar="NAME", help="Select a device (optional name, default: only list names)")
  g_profile.add_argument("--offset", type=int, default=0, metavar="N", help="event offset (default: 0)")
  g_profile.add_argument("--limit", type=int, default=10, metavar="N", help="events to display (-1 for all, default: 10)")
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
    profile = decode_profile(viz.get_profile(profile_data:=viz.load_pickle(args.profile_path, default=[])))
    viz.load_amd_counters(viz.ctxs, profile_data)
    counters = {f'{c["name"]} SQTT {s["name"]}': s["data"] for c in viz.ctxs if c["name"].startswith("Exec") for s in c["steps"]
                if s["name"].startswith("PKTS")}
    if args.device is None:
      print("Select a device:")
      for k in (*profile["layout"], *counters):
        print(f"  {k}")
      sys.exit(0)

    # SQTT printer
    if args.device in counters:
      sqtt_events = viz.sqtt_timeline(*counters[args.device])
      sqtt_pkts = [e for e in sqtt_events if type(e).__name__ == "ProfileRangeEvent"]
      pc_map = next(e.arg for e in sqtt_events if type(e).__name__ == "ProfilePointEvent" and e.key == 'pcMap')
      # modern terminals support 24-bit color
      def hex_colored(st:str, color:str) -> str: return f"\x1b[38;2;{int(color[1:3],16)};{int(color[3:5],16)};{int(color[5:7],16)}m{st}\x1b[0m"
      WAVE_COLORS = ((('VALU', 'VINTERP'), '#ffffc0'), (('SALU',), '#cef263'), (('VMEM',), '#b2b7c9'), (('LOAD', 'SMEM'), '#ffc0c0'),
                     (('STORE',), '#4fa3cc'), (('IMMEDIATE',), '#f3b44a'), (('BARRIER',), '#d00000'), (('LDS',), '#9fb4a6'), (('JUMP',), '#ffb703'),
                     (('JUMP_NO',), '#fb8500'), (('MESSAGE',), '#90dbf4'), (('WAVERDY',), '#1a2a2a'))
      total_sqtt_pkts = len(sqtt_pkts)
      start_idx, end_idx = args.offset, total_sqtt_pkts if args.limit == -1 else min(args.offset+args.limit, total_sqtt_pkts)
      print(f"{args.device} Instruction Trace:\n")
      print(f"{'#':<6} {'Clk':<11} {'Unit':<28} {'Op':<15} {'Dur':<4} {'Info'}")
      print("-" * 100)
      pkt_idxs:dict[str, itertools.count] = {}
      for e in sqtt_pkts[:start_idx]: next(pkt_idxs.setdefault(e.device, itertools.count()))
      for i, e in enumerate(sqtt_pkts[start_idx:end_idx], start=start_idx):
        op_name, info = e.name.display_name, e.name.ret or ""
        color = next((c for p, c in WAVE_COLORS if any(x in op_name for x in p)), None)
        op_str = hex_colored(op_name, color) if color else op_name
        if info.startswith("PC:"): info += f" {pc_map[int(info[3:])]}"
        pkt_unit = f"{e.device}-{next(pkt_idxs.setdefault(e.device, itertools.count()))}"
        print(f"{i:<6} {int(e.st):<11} {pkt_unit:<28} {op_str}{' '*(15-ansilen(op_str))} {int(e.en-e.st):<4} {info}")
      if args.limit != -1 and (start_idx > 0 or end_idx < total_sqtt_pkts):
        print(f"Showed events {start_idx}-{end_idx} from {total_sqtt_pkts} events, set --offset and --limit to see others")
      sys.exit(0)

    agg, total, n = {}, 0, 0
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
            print(f"{name} {ptm}/{(et or 0)*1e3:9.2f}ms  "+e.get('fmt', '').replace('\n', ' | ')+"  ")
            n += 1
        else:
          a = agg.setdefault(e["name"], [0.0, 0])
          a[0] += et
          a[1] += 1
          total += et
    if agg and total > 0:
      items = sorted(agg.items(), key=lambda kv:kv[1][0], reverse=True)
      sel = items if args.limit == -1 else items[args.offset:args.offset+args.limit]
      table = [[name, time_to_str(t, w=9), c, f"{(t/total*100.0):.2f}%"] for name,(t,c) in sel]
      if args.limit != -1 and (other:=items[len(sel):]):
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
