The following files contain C code (hopefully) equivalent to the AMX instructions:

|File|Contents|
|---|---|
|[emulate.h](emulate.h)|Common definitions and routines
|[ldst.c](ldst.c)|`ldx`, `ldy`, `ldz`, `ldzi`, `stx`, `sty`, `stz`, `stzi`|
|[extr.c](extr.c)|`extrx`, `extry`, `extrh`, `extrv`|
|[fma.c](fma.c)|`fma16`, `fma32`, `fma64`|
|[fms.c](fms.c)|`fms16`, `fms32`, `fms64`|
|[genlut.c](genlut.c)|`genlut`|
|[mac16.c](mac16.c)|`mac16`|
|[matfp.c](matfp.c)|`matfp`|
|[matint.c](matint.c)|`matint`|
|[vecfp.c](vecfp.c)|`vecfp`|
|[vecint.c](vecint.c)|`vecint`|

The file [`test.c`](test.c) contains a very simple test harness that generates a bunch of random operands, and asserts that the behaviour of the above files matches the behaviour of the AMX instructions. This test harness requires an Apple Silicon machine, as otherwise the AMX instructions are not available to test against.

A _very_ simple [`Makefile`](Makefile) is provided to compile all of the above. Running `make test` should output:
```
Testing AMX_LDX... OK   
Testing AMX_LDY... OK   
Testing AMX_LDZ... OK   
Testing AMX_LDZI... OK   
Testing AMX_STX... OK   
Testing AMX_STY... OK   
Testing AMX_STZ... OK   
Testing AMX_STZI... OK   
Testing AMX_EXTRX... OK   
Testing AMX_EXTRY... OK   
Testing AMX_MAC16... OK   
Testing AMX_FMA16... OK   
Testing AMX_FMA32... OK   
Testing AMX_FMA64... OK   
Testing AMX_FMS16... OK   
Testing AMX_FMS32... OK   
Testing AMX_FMS64... OK   
Testing AMX_VECINT... OK   
Testing AMX_VECFP... OK   
Testing AMX_MATINT... OK   
Testing AMX_MATFP... OK   
Testing AMX_GENLUT... OK
```
