from extra.assembly.amd.asm import get_dsl

with open("./extra/gemm/asm/gemm.s", "r") as f: text = f.read()

stream = []
for line in text.split("\n"):
  line = line.strip()
  if not line: continue
  if line.startswith("/"): continue
  if line.endswith(":"):
    stream.append(f'"{line}"')
    continue
  t = line.split("/")[0]
  if not t: continue
  dsl = get_dsl(t)
  stream.append(dsl)

with open("./extra/gemm/asm/gemm2.py", "w") as f:
  f.write("\n".join(stream))
