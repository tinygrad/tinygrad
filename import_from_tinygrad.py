#!/usr/bin/env python3
import os

# all the top level dir tinygrad files
FILES = ["tensor.py", "device.py", "dtype.py", "helpers.py", "gradient.py", "__init__.py"]

# ops
FILES = ["uop/ops.py"]

# renderer
FILES = ["renderer/__init__.py", "renderer/cstyle.py"]

# engine
FILES = ["engine/grouper.py", "engine/realize.py", "engine/schedule.py"]

# codegen
FILES = ["codegen/__init__.py", "codegen/devectorizer.py", "codegen/expander.py", "codegen/heuristic.py", "codegen/kernel.py",
         "codegen/linearize.py", "codegen/lowerer.py", "codegen/symbolic.py", "codegen/transcendental.py"]

# shape
FILES = ["shape/shapetracker.py", "shape/view.py"]

# runtime
FILES = ["runtime/ops_cpu.py", "runtime/ops_python.py", "runtime/ops_disk.py", "runtime/ops_metal.py", "runtime/support/elf.py"]

# runtime (to remove)
# TODO: ops_python shouldn't be needed, remove from tensor.py
# TODO: llvm shouldn't be needed, allow no import in metal
FILES = ["runtime/autogen/libc.py", "runtime/autogen/llvm.py", "runtime/support/llvm.py"]

# these are all in tinygrad/ folder
FILES = ["tinygrad/"+x for x in FILES]

# basic utils
FILES += ["ruff.toml", "sz.py"]

os.system("git checkout master "+' '.join(FILES))

