# tinygrad directory layout

Listed in order of how they are processed

---

## tinygrad/kernelize

Group UOps into kernels.

::: tinygrad.kernelize.kernelize.get_kernelize_map
    options:
        members: false
        show_labels: false
        show_source: false

---

## tinygrad/opt

Transforms the ast into an optimized ast. This is where BEAM search and heuristics live.

When finished, this will just have a function that takes in the ast and returns the optimized ast.

---

## tinygrad/codegen

Transform the optimized ast into a linearized list of UOps.

::: tinygrad.codegen.full_rewrite
    options:
        members: false
        show_labels: false
        show_source: false

---

## tinygrad/renderer

Transform the linearized list of UOps into a program.

---

## tinygrad/engine

Abstracted high level interface to the runtimes.
