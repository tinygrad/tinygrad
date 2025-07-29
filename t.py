import subprocess, pickle, pathlib, tempfile, os
from tinygrad.device import ProfileProgramEvent
from tinygrad.helpers import temp, OSX

with open("/Users/qazal/profile.pkl.qazal", "rb") as f: profile = pickle.load(f)
for e in profile:
  if isinstance(e, ProfileProgramEvent): break

RGA_PATH = [pathlib.Path(os.getenv("RGA_PATH", pathlib.Path.home()/"RadeonDeveloperToolSuite-2025-07-01-1408/rga.exe"))]
if OSX: RGA_PATH = ["wine"]+RGA_PATH
(out:=pathlib.Path(temp("rga_output"))).mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(delete=True) as elf:
  elf.write(e.lib)
  elf.flush()
  out = subprocess.check_output([*RGA_PATH, "-s", "bin", "--isa", out/"disassem.txt", "--parse-isa", "--livereg", out/"livereg.txt", "--line-numbers",
                                 "--analysis", out/"resourceUsage.csv", elf.name], text=True)
print(f"Saved to {out}")
