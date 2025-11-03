# load csv export output from ncu
import csv, json, sys

# temp for ncu csv export numeric data
def try_number(name:str, prev:str) -> int|float|str:
  x = prev.split(" ")[0]
  x = x.replace(",", "")
  num = None
  try: num = int(x)
  except ValueError:
    try: num = int(f) if (f:=float(x)).is_integer() else f
    except ValueError: return prev
  assert num is not None
  if "byte]" in name:
    x = name.split(" ")[1].split("byte")[0][-1].replace("[", "").strip()
    if x: num *= {"G":1e9, "M":1e6, "K":1e3}[x]
  return num

def load_custom(fp:str, ctxs:list[dict]):
  counters:list[dict] = []
  with open(fp) as f:
    reader = csv.DictReader(f)
    for row in reader:
      name, *rest = row.values()
      if not counters: counters = [{} for _ in range(len(rest))]
      for i,x in enumerate(rest): counters[i][name.split(" ")[0]] = try_number(name, x)
  steps = [{"name":x["Function"], "depth":0, "data":{"src":json.dumps(counters[i], indent=2), "device":"CUDA"},
            "query":f"/render?ctx={len(ctxs)}&step={i}&fmt=counters"} for i,x in enumerate(counters)]
  ctxs.append({"name":"Counters", "steps":steps})

if __name__ == "__main__":
  load_custom(sys.argv[1], cret:=[])
  for x in cret[0]["steps"]:
    print(x["name"])
    print(json.loads(x["data"]["src"]))
