# RDNA3 Assembly Toolkit

RDNA3.5 assembler, disassembler, and emulator for AMD GPUs. Parses instruction definitions directly from the AMD ISA PDF. Enables testing tinygrad's AMD backend without hardware.

## Module Structure

```
rdna3/
├── lib.py      # Core DSL: BitField, Reg types (SGPR/VGPR/TTMP), Inst base class
├── asm.py      # Assembler (asm) and disassembler (disasm)
├── emu.py      # Instruction emulator: WaveState, run_asm, exec_wave
├── gen.py      # Parses AMD ISA PDF -> autogen/__init__.py
├── pcode.py    # Compiles PDF pseudocode semantics to Python functions
├── autogen/
│   ├── __init__.py   # Generated: SrcEnum, opcodes, instruction formats
│   └── gen_pcode.py  # Generated: compiled pseudocode functions
└── test/       # Test suite
```

## Key Concepts

### Register Types
```python
from extra.assembly.rdna3.autogen import s, v, ttmp
s[0]        # SGPR 0
s[0:3]      # SGPR pair s0:s3 (4 registers)
v[5]        # VGPR 5
ttmp[0]     # Trap temporary register
```

### Instruction Encoding
- `Inst32`: 32-bit instructions (VOP1, VOP2, VOPC, SOP1, SOP2, SOPC, SOPK, SOPP)
- `Inst64`: 64-bit instructions (VOP3, VOP3SD, VOP3P, SMEM, FLAT, DS, MUBUF, etc.)
- Literal values append 4 bytes (8 bytes for 64-bit ops)

### Source Operand Encoding
- 0-105: SGPRs
- 106-107: VCC_LO, VCC_HI
- 108-123: TTMP registers
- 124: NULL
- 128-192: Inline constants 0-64
- 193-208: Inline constants -1 to -16
- 240-248: Float constants (0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 1/2pi)
- 255: Literal (next dword)
- 256-511: VGPRs

### VOP3 vs VOP3SD
VOP3SD (scalar destination) shares encoding with VOP3 but uses different field layout. Check opcode to distinguish:
```python
VOP3SD_OPCODES = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}
```

## Common Workflows

### Running Tests
```bash
# All tests
pytest extra/assembly/rdna3/test/ -x

# Specific test file
pytest extra/assembly/rdna3/test/test_emu.py -xvs

# Compare emulator with real hardware (requires AMD GPU)
USE_HW=1 pytest extra/assembly/rdna3/test/test_emu.py -xvs
```

### Regenerating Autogen
```bash
# Regenerate instruction formats from AMD ISA PDF
python extra/assembly/rdna3/gen.py

# Regenerate compiled pseudocode functions
python -m extra.assembly.rdna3.pcode
```

### Basic Usage
```python
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.asm import asm, disasm

# Assemble from text
inst = asm("v_add_f32_e32 v0, v1, v2")

# Or use instruction helpers directly
inst = v_add_f32_e32(v[0], v[1], v[2])

# Disassemble
print(disasm(inst))  # "v_add_f32_e32 v0, v1, v2"

# Get machine code
code = inst.to_bytes()

# Run in emulator
from extra.assembly.rdna3.emu import run_asm
run_asm(lib_ptr, lib_size, gx=1, gy=1, gz=1, lx=32, ly=1, lz=1, args_ptr=args)
```

## Test Files

| File | Purpose |
|------|---------|
| `test_emu.py` | Emulator instruction execution (USE_HW=1 for hardware comparison) |
| `test_roundtrip.py` | Assemble -> disassemble -> reassemble verification |
| `test_rdna3_asm.py` | Assembler syntax parsing |
| `test_llvm.py` | Compare against LLVM disassembler output |
| `test_pcode.py` | Pseudocode DSL and compilation |
| `test_formats.py` | Instruction format encoding/decoding |

## Adding New Instructions

1. **Opcode already in PDF**: Run `python extra/assembly/rdna3/gen.py` to regenerate autogen
2. **Pseudocode needed**: The PDF pseudocode is auto-compiled by `pcode.py`. Run `python -m extra.assembly.rdna3.pcode`
3. **Non-ALU ops** (memory, control flow): Add handling in `emu.py` directly
4. **Special semantics**: Add to `UNSUPPORTED` list in `pcode.py` and implement manually in `emu.py`

## Gotchas

### PDF Errors Fixed in gen.py
- SMEM: PDF says DLC=bit14, GLC=bit16 but actual encoding is DLC=bit13, GLC=bit14

### Inline Constants Differ by Type
```python
# f32: 0.5 -> 0x3f000000
# f16: 0.5 -> 0x3800 (in low 16 bits only, NOT replicated)
# f64: 0.5 -> 0x3fe0000000000000
```

### 64-bit Literals
Instructions ending in `_F64`, `_B64`, `_I64`, `_U64` use 8-byte literals instead of 4-byte.

### VOPD Dual-Issue
VOPD encodes two VOP operations. vdsty is encoded specially:
```python
vdsty_actual = (vdsty_encoded << 1) | ((vdstx & 1) ^ 1)
```

### Lane Instructions
`V_READFIRSTLANE_B32`, `V_READLANE_B32`, `V_WRITELANE_B32` require special handling - they access VGPRs across lanes and write to SGPRs.
