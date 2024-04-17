## Quick summary

|Instruction|General theme|Notes|
|---|---|---|
|`set`|Setup AMX state|Raises invalid instruction exception if already setup. All registers set to zero.|
|`clr`|Clear AMX state|All registers set to uninitialised, no longer need saving/restoring on context switch.|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|Must be `17`|
|0|5|5-bit immediate|`0` for `set`<br/>`1` for `clr`|

## Description

Before any other AMX instructions can be used, `set` must be issued. At some point later, `clr` must be issued. The pair of instructions are _not_ re-entrant: `set` will raise an invalid instruction exception if AMX is already enabled for the issuing thread. The implied ABI for public functions is therefore: AMX will be disabled on entry, and if enabled within the function, must be disabled again before returning.

As a side-effect of `set`, all X and Y and Z registers are set to zero. If the goal is merely setting Z to zero, then various computational instructions can be used instead, for example [`fma16`](fma.md) with bits 27=1, 28=1, 29=1, 62=1. Alternatively, some computational instructions that accumulate onto Z can be configured to read Z as zero (bit 27 to [`fma16`](fma.md) / [`fma32`](fma.md) / [`fma64`](fma.md) / [`fms16`](fms.md) / [`fms32`](fms.md) / [`fms64`](fms.md) / [`mac16`](mac16.md)).

## Emulation code

Do not require emulation
