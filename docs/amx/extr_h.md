## Quick summary

|Instruction|General theme|Writemask|Optional special features|
|---|---|---|---|
|`extrh`&nbsp;(26=0)|<code>x[i] =&nbsp;&nbsp;&nbsp;z[_][i]&nbsp;</code>|7 bit|
|`extrh`&nbsp;(26=1,10=0)|`x[i] = f(z[_][i])`|9 bit|Integer right shift, integer saturation|
|`extrh`&nbsp;(26=1,10=1)|`y[i] = f(z[_][i])`|9 bit|Integer right shift, integer saturation|

## Instruction encoding

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|10|22|[A64 reserved instruction](aarch64.md)|Must be `0x201000 >> 10`|
|5|5|Instruction|Must be `8`|
|0|5|5-bit GPR index|See below for the meaning of the 64 bits in the GPR|

## Operand bitfields when 26=1

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|63|1|Lane width mode (hi)|See bit 11|
|(63=1)&nbsp;62|1|Destination is bf16 (`1`) or f16 (`0`)|Only applies in mixed lane-width modes, ignored otherwise|
|(63=1)&nbsp;54|8|Ignored|Only applies in mixed lane-width modes, ignored otherwise|
|(63=0)&nbsp;58|5|Right shift amount|Only applies in mixed lane-width modes, ignored otherwise|
|(63=0)&nbsp;57|1|Z is signed (`1`) or unsigned (`0`)|Only applies in mixed lane-width modes, ignored otherwise|
|(63=0)&nbsp;56|1|Z saturation is signed (`1`) or unsigned (`0`)|Only applies in mixed lane-width modes, ignored otherwise|
|(63=0)&nbsp;55|1|Saturate Z (`1`) or truncate Z (`0`)|Only applies in mixed lane-width modes, ignored otherwise|
|(63=0)&nbsp;54|1|Right shift is rounding (`1`) or truncating (`0`)|Only applies in mixed lane-width modes, ignored otherwise|
|41|13|Ignored|
|(31=0)&nbsp;38|3|Write enable mode|
|(31=0)&nbsp;32|6|Write enable value|Meaning dependent upon associated mode|
|31|1|Perform operation for multiple vectors (`1`)<br/>or just one vector (`0`)|M2 only (always reads as `0` on M1)|
|27|4|Ignored|
|26|1|Must be `1` for this decode variant|
|(31=1)&nbsp;25|1|"Multiple" means four vectors (`1`)<br/>or two vectors (`0`)|Top two bits of Z row ignored if operating on four vectors|
|20|6|Z row|When 31=1, top bit or top two bits ignored|
|15|5|Ignored|
|11|4|Lane width mode (lo)|See bit 63|
|10|1|Destination is Y (`1`) or is X (`0`)|
|9|1|Ignored|
|0|9|Destination offset (in bytes)|

Lane widths:
|X (or Y)|Z|63|11|Notes|
|---|---|---|---|---|
|i8 or u8|i8 or u8|`0`|`0`|
|i32 or u32|i32 or u32|`0`|`8`|
|i16 or u16|i32 or u32 (two rows, interleaved pair)|`0`|`9`|Shift and saturation supported|
|i16 or u16|i32 or u32 (four rows, interleaved pair from those)|`0`|`10`|Shift and saturation supported|
|i8 or u8|i32 or u32 (four rows, interleaved quartet)|`0`|`11`|Shift and saturation supported|
|i8 or u8|i16 or u16 (two rows, interleaved pair)|`0`|`13`|Shift and saturation supported|
|i16 or u16|i16 or u16|`0`|anything else|
|f64|f64|`1`|`1`|
|f32|f32|`1`|`8`|
|f16 or bf16|f32 (two rows, interleaved pair)|`1`|`9`|M2 only. Bit 62 determines X (or Y) format|
|f16 or bf16|f32 (four rows, interleaved pair from those)|`1`|`10`|M2 only. Bit 62 determines X (or Y) format|
|f16|f16|`1`|anything else|

Write enable modes (with regard to X or Y):
|Mode|Meaning of value (N)|
|---:|---|
|`0`|Enable all lanes (`0` or `4` or `5`), or odd lanes only (`1`), or even lanes only (`2`), or enable all lanes but write `0` to them regardless of Z (`3`), or no lanes enabled (anything else)|
|`1`|Only enable lane #N|
|`2`|Only enable the first N lanes, or all lanes when N is zero|
|`3`|Only enable the last N lanes, or all lanes when N is zero|
|`4`|Only enable the first N lanes (no lanes when N is zero)|
|`5`|Only enable the last N lanes (no lanes when N is zero)|
|`6`|No lanes enabled|
|`7`|No lanes enabled|

## Operand bitfields when 26=0

|Bit|Width|Meaning|Notes|
|---:|---:|---|---|
|48|16|Ignored|
|46|2|Write enable mode||
|41|5|Write enable value|Meaning dependent upon associated mode|
|30|11|Ignored|
|28|2|Lane width mode|
|27|1|Must be `0`|Otherwise decodes as [`extrx`](extr_x.md)|
|26|1|Must be `0` for this decode variant|
|20|6|Z row|
|19|1|Ignored|
|10|9|Destination offset (in bytes)|Destination is always X for this decode variant|
|0|10|Ignored|

Lane width modes:
|X,Z|28|
|---|---|
|any 64-bit|`0`|
|any 32-bit|`1`|
|any 16-bit|`2`|
|any 16-bit, but with high 8 bits of each lane disabled|`3`|

Write enable modes (with regard to X):
|Mode|Meaning of value (N)|
|---:|---|
|`0`|Enable all lanes (`0`), or odd lanes only (`1`), or even lanes only (`2`), or no lanes (anything else)|
|`1`|Only enable lane #N|
|`2`|Only enable the first N lanes, or all lanes when N is zero|
|`3`|Only enable the last N lanes, or all lanes when N is zero|

## Description

When X/Y/Z all have the same lane width (which is always the case when 26=0), this operation is simple: the field at bit 20 identifies a Z row, and that row is copied to X (or transposed and copied to Y). The lane width only affects the write-enable logic.

When Z is wider than X/Y, this operation is more complex, as it needs to perform narrowing. The four mixed-width modes are `9`, `10`, `11`, `13`. For integer operands, all of these modes support right-shift and optional saturation of the Z values, and then take the low bits. For floating-point operands, these modes canonicalise NaNs and perform rounding (round to nearest, ties to even).

Mode 9 (32-bit Z elements, 16-bit X or Y elements), correspondance between X/Y lanes and pair of Z registers:
<table><tr><th>Z0</th><td>0</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td><td>14</td><td>16</td><td>18</td><td>20</td><td>22</td><td>24</td><td>26</td><td>28</td><td>30</td></tr>
<tr><th>Z1</th><td>1</td><td>3</td><td>5</td><td>7</td><td>9</td><td>11</td><td>13</td><td>15</td><td>17</td><td>19</td><td>21</td><td>23</td><td>25</td><td>27</td><td>29</td><td>31</td></tr></table>

Mode 10 (32-bit Z elements, 16-bit X/Y elements), correspondance between X/Y lanes and quartet of Z registers:
<table><tr><th>Z0</th><td>0</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td><td>14</td><td>16</td><td>18</td><td>20</td><td>22</td><td>24</td><td>26</td><td>28</td><td>30</td></tr>
<tr><th>Z1</th><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/></tr>
<tr><th>Z2</th><td>1</td><td>3</td><td>5</td><td>7</td><td>9</td><td>11</td><td>13</td><td>15</td><td>17</td><td>19</td><td>21</td><td>23</td><td>25</td><td>27</td><td>29</td><td>31</td></tr>
<tr><th>Z3</th><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/><td/></tr></table>

Mode 11 (32-bit Z elements, 8-bit X/Y elements), correspondance between X/Y lanes and quartet of Z registers:
<table><tr><th>Z0</th><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td><td>32</td><td>36</td><td>40</td><td>44</td><td>48</td><td>52</td><td>56</td><td>60</td></tr>
<tr><th>Z1</th><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td><td>33</td><td>37</td><td>41</td><td>45</td><td>49</td><td>53</td><td>57</td><td>61</td></tr>
<tr><th>Z2</th><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td><td>34</td><td>38</td><td>42</td><td>46</td><td>50</td><td>54</td><td>58</td><td>62</td></tr>
<tr><th>Z3</th><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td><td>35</td><td>39</td><td>43</td><td>47</td><td>51</td><td>55</td><td>59</td><td>63</td></tr></table>

Mode 13 (16-bit Z elements, 8-bit X/Y elements), correspondance between X/Y lanes and pair of Z registers:
<table><tr><th>Z0</th><td>0</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td><td>14</td><td>16</td><td>18</td><td>20</td><td>22</td><td>24</td><td>26</td><td>28</td><td>30</td><td>32</td><td>34</td><td>36</td><td>38</td><td>40</td><td>42</td><td>44</td><td>46</td><td>48</td><td>50</td><td>52</td><td>54</td><td>56</td><td>58</td><td>60</td><td>62</td></tr>
<tr><th>Z1</th><td>1</td><td>3</td><td>5</td><td>7</td><td>9</td><td>11</td><td>13</td><td>15</td><td>17</td><td>19</td><td>21</td><td>23</td><td>25</td><td>27</td><td>29</td><td>31</td><td>33</td><td>35</td><td>37</td><td>39</td><td>41</td><td>43</td><td>45</td><td>47</td><td>49</td><td>51</td><td>53</td><td>55</td><td>57</td><td>59</td><td>61</td><td>63</td></tr></table>

On M2, when 26=1, the whole operation can optionally be repeated multiple times, by setting bit 31. Bit 25 controls the repetition count; either two times or four times. Consecutive X or Y registers are used as the destination. If repeated twice, the top bit of Z row is ignored, and Z row is incremented by 32 for the 2<sup>nd</sup> iteration. If repeated four times, the top two bits of Z row are ignored, and Z row is incremented by 16 on each iteration.

## Emulation code

See [extr.c](extr.c).

A representative sample is:
```c
void emulate_AMX_EXTRX(amx_state* state, uint64_t operand) {
    void* dst;
    uint64_t dst_offset;
    uint64_t z_row = operand >> 20;
    uint64_t z_step = 64;
    uint64_t store_enable = ~(uint64_t)0;
    uint8_t buffer[64];
    uint32_t stride = 0;
    uint32_t zbytes, xybytes;

    if (operand & EXTR_HV) {
        dst = (operand & EXTR_HV_TO_Y) ? state->y : state->x;
        dst_offset = operand;
        switch (((operand >> 63) << 4) | ((operand >> 11) & 0xF)) {
        case  0: xybytes = 1; zbytes = 1; break;
        case  8: xybytes = 4; zbytes = 4; break;
        case  9: xybytes = 2; zbytes = 4; stride = 1; break;
        case 10: xybytes = 2; zbytes = 4; stride = 2; break;
        case 11: xybytes = 1; zbytes = 4; stride = 1; break;
        case 13: xybytes = 1; zbytes = 2; stride = 1; break;
        case 17: xybytes = 8; zbytes = 8; break;
        case 24: xybytes = 4; zbytes = 4; break;
        case 25: xybytes = 2; if (AMX_VER >= AMX_VER_M2) { zbytes = 4; stride = 1; } else { zbytes = 2; } break;
        case 26: xybytes = 2; if (AMX_VER >= AMX_VER_M2) { zbytes = 4; stride = 2; } else { zbytes = 2; } break;
        default: xybytes = 2; zbytes = 2; break;
        }
        if ((AMX_VER >= AMX_VER_M2) && (operand & (1ull << 31))) {
            operand &=~ (0x1ffull << 32);
            z_step = z_row & 32 ? 16 : 32;
        }
        store_enable &= parse_writemask(operand >> 32, xybytes, 9);
    } else if (operand & EXTR_BETWEEN_XY) {
        ...
    } else {
        dst = state->x;
        dst_offset = operand >> 10;
        xybytes = 8 >> ((operand >> 28) & 3);
        if (xybytes == 1) {
            xybytes = 2;
            store_enable &= 0x5555555555555555ull;
        }
        store_enable &= parse_writemask(operand >> 41, xybytes, 7);
        zbytes = xybytes;
    }

    uint32_t signext = (operand & EXTR_SIGNED_INPUT) ? 64 - zbytes*8 : 0;
    for (z_row &= z_step - 1; z_row <= 63; z_row += z_step) {
        for (uint32_t i = 0; i < 64; i += xybytes) {
            uint64_t zoff = (i & (zbytes - 1)) / xybytes * stride;
            int64_t val = load_int(&state->z[bit_select(z_row, z_row + zoff, zbytes - 1)].u8[i & -zbytes],
                                   zbytes, signext);
            if (stride) val = extr_alu(val, operand, xybytes*8);
            store_int(buffer + i, xybytes, val);
        }
        if ((operand & EXTR_HV) && (((operand >> 32) & 0x1ff) == 3)) {
            memset(buffer, 0, sizeof(buffer));
        }
        store_xy_row(dst, dst_offset & 0x1FF, buffer, store_enable);
        dst_offset += 64;
    }
}

int64_t extr_alu(int64_t val, uint64_t operand, uint32_t outbits) {
    uint32_t shift = (operand >> 58) & 0x1f;
    if (operand & (1ull << 63)) {
        if (shift >= 16) {
            val = bf16_from_f32((uint32_t)val);
        } else {
            __asm("fcvt %h0, %s0" : "=w"(val) : "0"(val));
        }
        return val;
    }
    if (shift && (operand & EXTR_ROUNDING_SHIFT)) {
        val += 1 << (shift - 1);
    }
    val >>= shift;
    if (operand & EXTR_SATURATE) {
        if (operand & EXTR_SIGNED_OUTPUT) outbits -= 1;
        int64_t hi = 1ull << outbits;
        if (operand & EXTR_SIGNED_INPUT) {
            int64_t lo = (operand & EXTR_SIGNED_OUTPUT) ? -hi : 0;
            if (val < lo) val = lo;
            if (val >= hi) val = hi - 1;
        } else {
            if ((uint64_t)val >= (uint64_t)hi) val = hi - 1;
        }
    }
    return val;
}
```
