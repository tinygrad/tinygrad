### Notation

`x` denotes a 64-byte span from the X register pool, accessed as a vector of lanes. The lanes are indexed by `i`.

`y` denotes a 64-byte span from the Y register pool, accessed as a vector of lanes. The lanes are indexed by `j` (or by `i` for vector operations).

`z` denotes the entire set of 64x64-byte Z registers, with 2D indexing. When only one index variable is used, `[_]` denotes that the other index comes from the instruction operand (typically a bitfield called "Z row" or "Z column").

`f` denotes _some_ function. `f(x, y) = x * y` is usually one option for binary functions. <code>f<sub>s</sub>(z) = z &gt;&gt; s</code> is usually one option for unary functions.

Some instructions can operate in multiple distinct modes. In these cases, the instruction name is followed by the relevant mode bits. When the mode field is a single bit #N, this is denoted as "(N=0)" or "(N=1)". When the mode field is multiple bits starting at bit #N, this is denoted as "(N=M)" or "(N≠M)" or "(N≤M)" or "(N≥M)".

## Setup and clear

|Instruction|General theme|Notes|
|---|---|---|
|[`set`](setclr.md)|Setup AMX state|Raises invalid instruction exception if already setup. All registers set to zero.|
|[`clr`](setclr.md)|Clear AMX state|All registers set to uninitialised, no longer need saving/restoring on context switch.|

## Memory

|Instruction|General theme|Optional special features|
|---|---|---|
|[`ldx`](ldst.md)|<code>&nbsp;&nbsp;&nbsp;x[i] = memory[i]</code>|Load pair|
|[`ldy`](ldst.md)|<code>&nbsp;&nbsp;&nbsp;y[i] = memory[i]</code>|Load pair|
|[`ldz`](ldst.md)<br/>[`ldzi`](ldst.md)|`z[_][i] = memory[i]`|Load pair, interleaved Z|
|[`stx`](ldst.md)|<code>memory[i] =&nbsp;&nbsp;&nbsp;&nbsp;x[i]</code>|Store pair|
|[`sty`](ldst.md)|<code>memory[i] =&nbsp;&nbsp;&nbsp;&nbsp;y[i]</code>|Store pair|
|[`stz`](ldst.md)<br/>[`stzi`](ldst.md)|`memory[i] = z[_][i]`|Store pair, interleaved Z|

## Floating-point matrix arithmetic (i.e. outer products), writing to `z`

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|[`fma64`&nbsp;(63=0)](fma.md)<br/>[`fma32`&nbsp;(63=0)](fma.md)<br/>[`fma16`&nbsp;(63=0)](fma.md)|`z[j][i] += x[i] * y[j]`|7 bit X, 7 bit Y|X/Y/Z input disable|
|[`fms64`&nbsp;(63=0)](fms.md)<br/>[`fms32`&nbsp;(63=0)](fms.md)<br/>[`fms16`&nbsp;(63=0)](fms.md)|`z[j][i] -= x[i] * y[j]`|7 bit X, 7 bit Y|X/Y/Z input disable|
|[`matfp`](matfp.md)|<code>z[j][i]&nbsp;±=&nbsp;f(x[i],&nbsp;y[j])</code>|9 bit X, 9 bit Y|Indexed X or Y, shuffle X, shuffle Y,<br/>positive selection|

## Integer matrix arithmetic (i.e. outer products), writing to `z`

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|[`mac16`&nbsp;(63=0)](mac16.md)|`z[j][i] += x[i] * y[j]`|7 bit X, 7 bit Y|X/Y/Z input disable, right shift|
|[`matint`&nbsp;(47≠4)](matint.md)|<code>z[j][i]&nbsp;±=&nbsp;f(x[i],&nbsp;y[j])</code>|9 bit X or Y|Indexed X or Y, shuffle X, shuffle Y,<br/>right shift, `sqrdmlah`, `popcnt`|
|[`matint`&nbsp;(47=4)](matint.md)|<code>z[j][i]&nbsp;&nbsp;=&nbsp;f(z[j][i])</code>|9 bit X or Y|Right shift, saturation|

## Floating-point vector arithmetic (i.e. pointwise products), writing to `z`

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|[`fma64`&nbsp;(63=1)](fma.md)<br/>[`fma32`&nbsp;(63=1)](fma.md)<br/>[`fma16`&nbsp;(63=1)](fma.md)|`z[_][i] += x[i] * y[i]`|7 bit|X/Y/Z input disable|
|[`fms64`&nbsp;(63=1)](fms.md)<br/>[`fms32`&nbsp;(63=1)](fms.md)<br/>[`fms16`&nbsp;(63=1)](fms.md)|`z[_][i] -= x[i] * y[i]`|7 bit|X/Y/Z input disable|
|[`vecfp`](vecfp.md)|<code>z[_][i]&nbsp;±=&nbsp;f(x[i],&nbsp;y[i])</code>|9 bit|Indexed X or Y, shuffle X, shuffle Y,<br/>broadcast Y element,<br/>positive selection, `min`, `max`|

## Integer vector arithmetic (i.e. pointwise products), writing to `z`

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|[`mac16`&nbsp;(63=1)](mac16.md)|<code>z[_][i]&nbsp;+=&nbsp;x[i]&nbsp;*&nbsp;y[i]</code>|7 bit|X/Y/Z input disable, right shift|
|[`vecint`&nbsp;(47≠4)](vecint.md)|<code>z[_][i]&nbsp;±=&nbsp;f(x[i],&nbsp;y[i])</code>|9 bit|Indexed X or Y, shuffle X, shuffle Y,<br/>broadcast Y element, right shift, `sqrdmlah`|
|[`vecint`&nbsp;(47=4)](vecint.md)|<code>z[\_][i]&nbsp;=&nbsp;f(z[\_][i])</code>|9 bit|Right shift, saturation|

## Vector data movement, writing to `x` or `y`

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|[`extrx`](extr_x.md)|`x[i] = y[i]`|None|
|[`extry`](extr_y.md)|`y[i] = x[i]`|None|
|[`extrh`&nbsp;(26=0)](extr_h.md)|<code>x[i] =&nbsp;&nbsp;&nbsp;z[_][i]&nbsp;</code>|7 bit|
|[`extrh`&nbsp;(26=1,10=0)](extr_h.md)|`x[i] = f(z[_][i])`|9 bit|Integer right shift, integer saturation|
|[`extrv`&nbsp;(26=1,10=0)](extr_v.md)|`x[j] = f(z[j][_])`|9 bit|Integer right shift, integer saturation|
|[`extrv`&nbsp;(26=0)](extr_v.md)|<code>y[j] =&nbsp;&nbsp;&nbsp;z[j][_]&nbsp;</code>|7 bit|
|[`extrv`&nbsp;(26=1,10=1)](extr_v.md)|`y[j] = f(z[j][_])`|9 bit|Integer right shift, integer saturation|
|[`extrh`&nbsp;(26=1,10=1)](extr_h.md)|`y[i] = f(z[_][i])`|9 bit|Integer right shift, integer saturation|

## Vector other

|Instruction|General theme|Notes|
|---|---|---|
|[`genlut`&nbsp;(53≤6)](genlut.md)|Generate indices for indexed load|For use by `matfp` / `matint` / `vecfp` / `vecint` / `genlut`&nbsp;(53≥7)|
|[`genlut`&nbsp;(53≥7)](genlut.md)|Perform indexed load|Can write to any of `x` or `y` or `z`|
