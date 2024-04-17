Assorted internet references:
* [dougallj/aarch64_amx.py](https://gist.github.com/dougallj/7a75a3be1ec69ca550e7c36dc75e0d6f) - IDA (disassembler) and Hex-Rays (decompiler) plugin for Apple AMX
* [dougallj/amx.h](https://gist.github.com/dougallj/7cba721da1a94da725ee37c1e9cd1f21) - amx simulator and hardware tests
* [amx-rs](https://github.com/yvt/amx-rs) - Rust bindings and simulator, includes ASM trick and genlut framework
* https://www.realworldtech.com/forum/?threadid=187087&curpostid=187120 - Instruction names
* https://news.ycombinator.com/item?id=24464807#24472004 - Instruction names
* [include/mach/arm/_structs.h](https://github.com/xybp888/iOS-SDKs/blob/a110a31ce82e42621b3e7ba31bd6563c02d2631a/iPhoneOS13.0.sdk/usr/include/mach/arm/_structs.h#L482) - Register file
* https://nod.ai/comparing-apple-m1-with-amx2-m1-with-neon/ - "AMX2"
* Google for `AMX_STATE_T_EL1`

Patents:
|Number|Title|Expires
|---|---|---|
|[US20180074824A1](https://patents.google.com/patent/US20180074824A1/en)|Outer Product Engine|Abandoned|
|[US10831488B1](https://patents.google.com/patent/US10831488B1/en)|Computation engine with extract instructions to minimize memory access|2038-10-31|
|[US20190129719A1](https://patents.google.com/patent/US20190129719A1/en)|Matrix Computation Engine (2017)|2038-01-14|
|[US10592239B2](https://patents.google.com/patent/US10592239B2/en)|Matrix computation engine (2019)|2037-11-01 (anticipated)|
|[US10877754B2](https://patents.google.com/patent/US10877754B2/en)|Matrix computation engine (2020)|2037-11-01 (anticipated)|
|[US10754649B2](https://patents.google.com/patent/US10754649B2/en)|Computation engine that operates in matrix and vector modes (2018)|2038-10-19|
|[US11042373B2](https://patents.google.com/patent/US11042373B2/en)|Computation engine that operates in matrix and vector modes (2020)|2038-07-24 (anticipated)|
|[US10970078B2](https://patents.google.com/patent/US10970078B2/en)|Computation engine with upsize/interleave and downsize/deinterleave options|2038-05-25|
|[US10642620B2](https://patents.google.com/patent/US10642620B2/en)|Computation engine with strided dot product (2018)|2038-07-24|
|[US10990401B2](https://patents.google.com/patent/US10990401B2/en)|Computation engine with strided dot product (2020)|2038-04-05 (anticipated)|

Academic papers:
* [Fast polynomial multiplication using matrix multiplication accelerators with applications to NTRU on Apple M1/M3 SoCs](https://eprint.iacr.org/2024/2) - "Our work repurposes AMX to implement polynomial multiplication and applies it to the NTRU cryptosystem, setting new speed records on the Apple M1 and M3 systems-on-chip (SoCs)."
