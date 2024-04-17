## Synopsis

```c
union amx_reg { // 64 byte vector...
    // ...of unsigned integers:
    uint8_t  u8 [64];
    uint16_t u16[32];
    uint32_t u32[16];
    // ...of signed integers:
    int8_t   i8 [64];
    int16_t  i16[32];
    int32_t  i32[16];
    // ...of IEEE 754 floating point:
    _Float16 f16[32]; // NB: IEEE half-precision, _not_ BF16
    float    f32[16];
    double   f64[ 8];
};

struct amx_state {
    amx_reg x[ 8]; // 512 bytes, of which 64 bytes extracted / inserted by operations
    amx_reg y[ 8]; // 512 bytes, of which 64 bytes extracted / inserted by operations
    amx_reg z[64]; // 64 by 64 matrix of bytes
}; // 5KB total
```

## Description

Each register is 64 bytes, viewed as vector of u8/u16/u32/i8/i16/i32/f16/f32/f64 elements. The architectural state contains 80 such registers: 8 of which in the X pool, 8 of which in the Y pool, and the remaining 64 forming a 64x64 grid called Z.

The entire X register pool can be concatenated to form a circular buffer of 512 bytes. Most instructions can operate on _any_ contiguous 64 byte range from this circular buffer. The same is true for Y: the entire Y pool can be concatenated to form a circular buffer of 512 bytes, and most instructions can operate on _any_ contiguous 64 byte range from this circular buffer.

Once 64 bytes of X and 64 bytes of Y have been selected, operations between X and Y and Z can be performed. Said operations fall into two main categories:
- Vector: Select one register from Z, and combine X/Y/Z in a standard SIMD manner: `Z[i] += X[i] * Y[i]`
- Matrix: Select a number of registers from Z equal to the number of lanes in X and Y, and combine X/Y/Z in an outer-product manner: `Z[j][i] += X[i] * Y[j]`

## Getting data in to and out of the AMX registers

Load/store instructions move data between memory and AMX registers.

Computation instructions can be used to synthesise various constants in the AMX registers: `0` is easy, as is floating-point `-0`. The latter can be used with integer shift instructions to synthesise (positive or negative) integer powers of two.

There is no direct movement between AMX registers and A64 general purpose registers or SIMD registers; data has to go via memory.

## Indexed loads

By default, instructions operate on a 64-byte span from X or Y. Some operations support _indexed loads_ rather than 64-byte span loads. Said loads are parameterised by two things: the element size and the index size. The element size (`ES`) is 8/16/32/64 bits, and the index size (`IS`) is 2/4/5 bits. The element _count_ (`EC`) is then 512 divided by the element size. A regular load would load an `ES * EC` (i.e. 512) bit span from X or Y. An indexed load instead loads an `IS * EC` bit span from X or Y, and then treats every group of `IS` bits as a lane index into a _different_ register with element size `ES`. For example, taking `ES` of 16 for f16 data and `IS` of 2, a 64-bit span is loaded from X or Y, which can be viewed as `u2[32]` vector, and is expanded to form an `f16[32]` vector by looking up into lanes 0/1/2/3 of some other `f16[32]` vector.

## Shuffles

Once a 64 byte X (or Y) vector has been obtained (either by a regular load or an indexed load), some instructions support shuffling the 64 bytes before use.

For vectors of 8 elements (i.e. `f64[8]`), the four (albeit only three distinct) available shuffles are:
<table>
<tr><th/><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th></tr>
<tr><th>S0</th><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr>
<tr><th>S1</th><td>0</td><td>4</td><td>1</td><td>5</td><td>2</td><td>6</td><td>3</td><td>7</td></tr>
<tr><th>S2</th><td>0</td><td>2</td><td>4</td><td>6</td><td>1</td><td>3</td><td>5</td><td>7</td></tr>
<tr><th>S3</th><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr>
</table>

For vectors of 16 elements (i.e. `f32[16]` or `i32[16]` or `u32[16]`), the four available shuffles are:
<table>
<tr><th/><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>10</th><th>11</th><th>12</th><th>13</th><th>14</th><th>15</th></tr>
<tr><th>S0</th><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr>
<tr><th>S1</th><td>0</td><td>8</td><td>1</td><td>9</td><td>2</td><td>10</td><td>3</td><td>11</td><td>4</td><td>12</td><td>5</td><td>13</td><td>6</td><td>14</td><td>7</td><td>15</td></tr>
<tr><th>S2</th><td>0</td><td>4</td><td>8</td><td>12</td><td>1</td><td>5</td><td>9</td><td>13</td><td>2</td><td>6</td><td>10</td><td>14</td><td>3</td><td>7</td><td>11</td><td>15</td></tr>
<tr><th>S3</th><td>0</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td><td>14</td><td>1</td><td>3</td><td>5</td><td>7</td><td>9</td><td>11</td><td>13</td><td>15</td></tr>
</table>

For vectors of 32 elements (i.e. `f16[32]` or `i16[32]` or `u16[32]`), the four available shuffles are:
<table>
<tr><th/><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>10</th><th>11</th><th>12</th><th>13</th><th>14</th><th>15</th><th>16</th><th>17</th><th>18</th><th>19</th><th>20</th><th>21</th><th>22</th><th>23</th><th>24</th><th>25</th><th>26</th><th>27</th><th>28</th><th>29</th><th>30</th><th>31</th></tr>
<tr><th>S0</th><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>22</td><td>23</td><td>24</td><td>25</td><td>26</td><td>27</td><td>28</td><td>29</td><td>30</td><td>31</td></tr>
<tr><th>S1</th><td>0</td><td>16</td><td>1</td><td>17</td><td>2</td><td>18</td><td>3</td><td>19</td><td>4</td><td>20</td><td>5</td><td>21</td><td>6</td><td>22</td><td>7</td><td>23</td><td>8</td><td>24</td><td>9</td><td>25</td><td>10</td><td>26</td><td>11</td><td>27</td><td>12</td><td>28</td><td>13</td><td>29</td><td>14</td><td>30</td><td>15</td><td>31</td></tr>
<tr><th>S2</th><td>0</td><td>8</td><td>16</td><td>24</td><td>1</td><td>9</td><td>17</td><td>25</td><td>2</td><td>10</td><td>18</td><td>26</td><td>3</td><td>11</td><td>19</td><td>27</td><td>4</td><td>12</td><td>20</td><td>28</td><td>5</td><td>13</td><td>21</td><td>29</td><td>6</td><td>14</td><td>22</td><td>30</td><td>7</td><td>15</td><td>23</td><td>31</td></tr>
<tr><th>S3</th><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr>
</table>

For vectors of 64 elements (i.e. `i8[64]` or `u8[64]`), the four available shuffles are:
<table>
<tr><th/><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>10</th><th>11</th><th>12</th><th>13</th><th>14</th><th>15</th><th>16</th><th>17</th><th>18</th><th>19</th><th>20</th><th>21</th><th>22</th><th>23</th><th>24</th><th>25</th><th>26</th><th>27</th><th>28</th><th>29</th><th>30</th><th>31</th><th>32</th><th>33</th><th>34</th><th>35</th><th>36</th><th>37</th><th>38</th><th>39</th><th>40</th><th>41</th><th>42</th><th>43</th><th>44</th><th>45</th><th>46</th><th>47</th><th>48</th><th>49</th><th>50</th><th>51</th><th>52</th><th>53</th><th>54</th><th>55</th><th>56</th><th>57</th><th>58</th><th>59</th><th>60</th><th>61</th><th>62</th><th>63</th></tr>
<tr><th>S0</th><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>22</td><td>23</td><td>24</td><td>25</td><td>26</td><td>27</td><td>28</td><td>29</td><td>30</td><td>31</td><td>32</td><td>33</td><td>34</td><td>35</td><td>36</td><td>37</td><td>38</td><td>39</td><td>40</td><td>41</td><td>42</td><td>43</td><td>44</td><td>45</td><td>46</td><td>47</td><td>48</td><td>49</td><td>50</td><td>51</td><td>52</td><td>53</td><td>54</td><td>55</td><td>56</td><td>57</td><td>58</td><td>59</td><td>60</td><td>61</td><td>62</td><td>63</td></tr>
<tr><th>S1</th><td>0</td><td>32</td><td>1</td><td>33</td><td>2</td><td>34</td><td>3</td><td>35</td><td>4</td><td>36</td><td>5</td><td>37</td><td>6</td><td>38</td><td>7</td><td>39</td><td>8</td><td>40</td><td>9</td><td>41</td><td>10</td><td>42</td><td>11</td><td>43</td><td>12</td><td>44</td><td>13</td><td>45</td><td>14</td><td>46</td><td>15</td><td>47</td><td>16</td><td>48</td><td>17</td><td>49</td><td>18</td><td>50</td><td>19</td><td>51</td><td>20</td><td>52</td><td>21</td><td>53</td><td>22</td><td>54</td><td>23</td><td>55</td><td>24</td><td>56</td><td>25</td><td>57</td><td>26</td><td>58</td><td>27</td><td>59</td><td>28</td><td>60</td><td>29</td><td>61</td><td>30</td><td>62</td><td>31</td><td>63</td></tr>
<tr><th>S2</th><td>0</td><td>16</td><td>32</td><td>48</td><td>1</td><td>17</td><td>33</td><td>49</td><td>2</td><td>18</td><td>34</td><td>50</td><td>3</td><td>19</td><td>35</td><td>51</td><td>4</td><td>20</td><td>36</td><td>52</td><td>5</td><td>21</td><td>37</td><td>53</td><td>6</td><td>22</td><td>38</td><td>54</td><td>7</td><td>23</td><td>39</td><td>55</td><td>8</td><td>24</td><td>40</td><td>56</td><td>9</td><td>25</td><td>41</td><td>57</td><td>10</td><td>26</td><td>42</td><td>58</td><td>11</td><td>27</td><td>43</td><td>59</td><td>12</td><td>28</td><td>44</td><td>60</td><td>13</td><td>29</td><td>45</td><td>61</td><td>14</td><td>30</td><td>46</td><td>62</td><td>15</td><td>31</td><td>47</td><td>63</td></tr>
<tr><th>S3</th><td>0</td><td>8</td><td>16</td><td>24</td><td>32</td><td>40</td><td>48</td><td>56</td><td>1</td><td>9</td><td>17</td><td>25</td><td>33</td><td>41</td><td>49</td><td>57</td><td>2</td><td>10</td><td>18</td><td>26</td><td>34</td><td>42</td><td>50</td><td>58</td><td>3</td><td>11</td><td>19</td><td>27</td><td>35</td><td>43</td><td>51</td><td>59</td><td>4</td><td>12</td><td>20</td><td>28</td><td>36</td><td>44</td><td>52</td><td>60</td><td>5</td><td>13</td><td>21</td><td>29</td><td>37</td><td>45</td><td>53</td><td>61</td><td>6</td><td>14</td><td>22</td><td>30</td><td>38</td><td>46</td><td>54</td><td>62</td><td>7</td><td>15</td><td>23</td><td>31</td><td>39</td><td>47</td><td>55</td><td>63</td></tr>
</table>

In all cases, S0 is the identity, S1 moves lane 1 to lane 2, S2 moves lane 1 to lane 4, and S3 moves lane 1 to lane 8.

## Per-byte write-enable

Most instructions support writing to only a subset of the output lanes, leaving the other lanes unchanged. This is controlled by a combination of a mode field and a value field. Said fields typically combine along the lines of:

|Mode|Meaning of value (N)|
|---:|---|
|`0`|Write to all lanes (`0`), or to odd lanes only (`1`), or to even lanes only (`2`), or to no lanes |
|`1`|Only write lane #N (or for certain vector operations, write all lanes, but broadcast Y lane #N to all lanes of Y)|
|`2`|Only write first N lanes, or to all lanes when N is zero|
|`3`|Only write last N lanes, or to all lanes when N is zero|
|`4`|Only write first N lanes (no lanes when N is zero)|
|`5`|Only write last N lanes (no lanes when N is zero)|
|`6`|Write to no lanes|
|`7`|Write to no lanes|

Matrix operations have separate write-enable for the X axis and the Y axis, with the enabled Z elements being the outer product of the two write-enables.

## Mixed lane widths

When the element size is identical between X and Y and Z, indexing is simple. Assume an element size in bits (ES) of 8, 16, 32, or 64 for all three, then X and Y have N elements, where N = 512 / ES. In vector mode, a single Z register also has N elements. In matrix mode, a 2D grid of N<sup>2</sup> values is used from Z: N distinct registers from Z, each containing N elements. The N distinct registers are equally spaced in the Y dimension, with spacing 64 / N (the user can choose the starting row, subject to 0 ≤ starting row < 64 / N).

When the element sizes are mixed (for example f16 × f16 ↦ f32 or i8 × i16 ↦ i32), then things are more complex. Either more Z registers need to be used (to make space for all the outputs), or some lanes from X and/or Y need to be dropped (because otherwise there is not space for all the outputs), or a combination of both. When lanes are dropped, it is typical to keep just the even lanes, or keep just one lane from every four (i.e. keep lanes 0, 4, 8, etc). Shuffles can be used to select different lanes; for example after applying shuffle S1 and then keeping just the even lanes, the result is lanes 0, 1, 2, etc; and after applying shuffle S2 and then keeping just one lane from every four then the result is lanes 0, 1, 2, etc. Alternatively, byte offsets on the input operands can be used to select different lanes: adding a byte offset equal to one lane turns even lanes into odd lanes, and turns lanes 0, 4, 8, etc into 1, 5, 9, etc.

One particularly common mixed-width combination is X and Y having element size of 16 bits (i.e. i16 or u32 or f16) and Z having element size 32 bits (i.e. i32 or u32 or f32). In this case, both X and Y have 32 elements, and every Z register has 16 elements. The complete outer product of X and Y would need 32<sup>2</sup> Z values, which there is _just_ space for: use all 64 Z registers, with 16 elements in each. Each 4 by 4 block of bytes ends up looking like:

<table><tr><td/><td colspan="2">X<sub>0:1</sub></td><td colspan="2">X<sub>2:3</sub></td></tr>
<tr><td rowspan="2">Y<sub>0:1</sub></td><td colspan="4">Z<sub>0,0:3</sub> += X<sub>0:1</sub> × Y<sub>0:1</sub></tr>
<tr><td colspan="4">Z<sub>1,0:3</sub> += X<sub>2:3</sub> × Y<sub>0:1</sub></td>
<tr><td rowspan="2">Y<sub>2:3</sub></td><td colspan="4">Z<sub>2,0:3</sub> += X<sub>0:1</sub> × Y<sub>2:3</sub></tr>
<tr><td colspan="4">Z<sub>3,0:3</sub> += X<sub>2:3</sub> × Y<sub>2:3</sub></td>
</table>

An alternative way of viewing this combination is that every _pair_ of Z registers contains 32 lanes (corresponding to the lanes of X), and there are 32 such _pairs_ (corresponding to the lanes of Y), with each pair arranged as:
<table><tr><th>Z0</th><td>0</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td><td>14</td><td>16</td><td>18</td><td>20</td><td>22</td><td>24</td><td>26</td><td>28</td><td>30</td></tr>
<tr><th>Z1</th><td>1</td><td>3</td><td>5</td><td>7</td><td>9</td><td>11</td><td>13</td><td>15</td><td>17</td><td>19</td><td>21</td><td>23</td><td>25</td><td>27</td><td>29</td><td>31</td></tr></table>

This arrangement is called an interleaved pair of Z registers, and for (16,16,32) has support instructions in the form of [`ldzi` and `stzi`](ldst.md#description).
