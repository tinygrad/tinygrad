# UOps (Micro-Operations)

The `tinygrad/uop/` directory defines the Micro-Operations (UOps), which are the intermediate representation (IR) of tinygrad.

## `ops.py`

This file defines the `UOp` class and the `Ops` enum.

### `Ops` Enum
The `Ops` enum defines all supported operations, categorized into:
- **ALU**: Arithmetic and logical operations (`ADD`, `MUL`, `MAX`, `CMPLT`, etc.).
- **Load/Store**: Memory access (`LOAD`, `STORE`).
- **Movement**: Reshaping and permutation (`RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP`).
- **Control Flow**: `RANGE`, `IF`, `LOOP`, `END`.
- **Special**: `DEFINE_GLOBAL`, `DEFINE_LOCAL`, `DEFINE_VAR`, `CONST`.
- **Buffer**: `BUFFER`, `BUFFER_VIEW`.
- **Other**: `CAST`, `BITCAST`, `VECTORIZE`, `GEP`, `PHI`.

### `UOp` Class
The `UOp` class represents a node in the computation graph.
- **Attributes**: `op` (type), `dtype` (data type), `src` (inputs), `arg` (arguments).
- **Methods**:
    - `replace()`: Returns a new UOp with modified attributes.
    - `toposort()`: Returns a topologically sorted list of the graph.
    - `substitute()`: Replaces UOps in the graph based on a dictionary.
    - `simplify()`: Simplifies the UOp graph using rewrite rules.
    - `render()`: Returns a string representation of the UOp (e.g., C code) using a pattern matcher.

### `PatternMatcher` and `UPat`
- **`UPat`**: Describes a pattern to match in the UOp graph. It can match op types, data types, arguments, and source patterns.
- **`PatternMatcher`**: A collection of rewrite rules (pattern -> replacement function).
- **`graph_rewrite`**: Applies a `PatternMatcher` to a UOp graph.

## `symbolic.py`

(Merged into `ops.py` in recent versions, but logic remains)
Handles symbolic math simplification using `graph_rewrite` and `PatternMatcher`.
- Implements simplification rules for arithmetic operations (e.g., `x + 0 -> x`, `x * 1 -> x`, constant folding).

## `spec.py`

Defines the specification and verification logic for UOps.

## `validate.py`

Contains validation logic for UOps.
