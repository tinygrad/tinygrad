"""Compatibility imports for the split A630 IR3 emulator.

New code should import decoder, register, or execution helpers from their owning
module.  This facade keeps the mock driver and existing focused tests stable.
"""
from test.mockgpu.qcom.decoder import IR3Instruction, decode_ir3
from test.mockgpu.qcom.dispatch import execute_dispatch
from test.mockgpu.qcom.executor import execute_ir3
from test.mockgpu.qcom.registers import local_id_regs, workgroup_id_regs

__all__ = ["IR3Instruction", "decode_ir3", "execute_dispatch", "execute_ir3", "local_id_regs", "workgroup_id_regs"]
