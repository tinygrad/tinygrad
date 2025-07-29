import subprocess, pickle, pathlib, tempfile, os
from tinygrad.device import ProfileProgramEvent
from tinygrad.helpers import temp, OSX

with open("/tmp/profile.pkl.qazal", "rb") as f: profile = pickle.load(f)
for e in profile:
  if isinstance(e, ProfileProgramEvent): break

RGA_PATH = os.getenv("RGA_PATH", str(pathlib.Path.home() / "RadeonDeveloperToolSuite-2025-07-01-1408" / "rga"))
RGA_BIN = ["wine", f"{RGA_PATH}.exe"] if OSX else [pathlib.Path(RGA_PATH)]
(out:=pathlib.Path(temp("rga_output"))).mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(delete=True) as elf:
  elf.write(e.lib)
  elf.flush()
  out = subprocess.check_output([*RGA_BIN, "-s", "bin", "--isa", out/"disassem.txt", "--parse-isa", "--livereg", out/"livereg.txt", "--line-numbers",
                                 "--analysis", out/"resourceUsage.csv", elf.name], text=True)
print(f"Saved to {out}")
