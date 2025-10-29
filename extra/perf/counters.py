# load csv export output from ncu
import csv, json

# temp for ncu csv export numeric data
def try_number(prev:str) -> int|float|str:
  x = prev.split(" ")[0]
  x = x.replace(",", "")
  try: return int(x)
  except ValueError:
    try: return int(f) if (f:=float(x)).is_integer() else f
    except ValueError: return prev

def load_custom(fp:str, ctxs:list[dict]):
  counters:list[dict] = []
  with open(fp) as f:
    reader = csv.DictReader(f)
    for row in reader:
      name, *rest = row.values()
      if not counters: counters = [{} for _ in range(len(rest))]
      for i,x in enumerate(rest): counters[i][name] = x
  steps = [{"name":x["Function Name"], "depth":0, "data":{"src":json.dumps(counters[i], indent=2), "lang":"txt", "device":"CUDA"},
            "query":f"/render?ctx={len(ctxs)}&step={i}&fmt=counters"} for i,x in enumerate(counters)]
  ctxs.append({"name":"Counters", "steps":steps})
