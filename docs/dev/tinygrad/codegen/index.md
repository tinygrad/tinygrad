# Codegen

The `tinygrad/codegen/` directory deals with generating kernel code from UOps.

## `__init__.py`

Exports kernel generation functions.
- **`get_program`**: Takes a `UOp` (sink) and a `Renderer`, and generates a `ProgramSpec` containing the source code.

## `simplify.py`

Implements simplification passes on the UOp graph before code generation.

## `gpudims.py`

Logic for calculating optimal GPU launch dimensions (global/local sizes).

## `opt/`

Contains optimization passes.
- **`search.py`**: Beam search or other search algorithms to find optimal kernel configurations (hand-tuning or auto-tuning).
- **`tc.py`**: Tensor Core usage logic.

## `late/`

Late-stage transformations (e.g., linearizing the graph for printing).
