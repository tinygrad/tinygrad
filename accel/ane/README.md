# The Apple Neural Engine

The Apple Neural Engine is a fancy DMA Engine that is based around convolutions. We don't have all the details worked out yet, but we can do some things with it. At its core, it runs through 0x300 ops in an hwx file. See `aneregs` for the registers used in each op.

It operates out of RAM or its 4MB L2 cache. The L2 "cache" appears to be manually managed, and only applies to the input and output, not the weights. The weights are usually included in the program, and it's unclear where they are copied to.

The 16 cores likely refer to the 16 wide Kernel DMA engine. They claim 11 TOPS total, which would be 687.5 GOPS/core. Perhaps it's a 32x32 MAC running at 335 MHz. That clock speed matches the cycle count time ratio from the debug perf stats.

It works with 5D Tensors, you specify the stride for the latter 4. All strides must be a multiple of 0x40 bytes
* Column (width)    -- aneRegs.Common.InDim.Win / aneRegs.Common.OutDim.Wout
* Row    (height)   -- aneRegs.Common.InDim.Hin / aneRegs.Common.OutDim.Hout
* Plane  (channels) -- aneRegs.Common.Cin.Cin / aneRegs.Common.Cout.Cout
* Depth
* Group  (batch)    -- aneRegs.Common.GroupConvCfg.NumGroups

It works with 3 data types
* UInt8
* Int8
* Float16

The ops have several parts
* Header -- The base addresses for the DMA engines
* KernelDMASrc -- 16x wide DMA engine for the weights/bias/nonlinearity
* Common -- Specifies the parameters for the convolution
* TileDMASrc -- Input DMA engine
* L2 -- Use the L2 cache for Source/Result instead of RAM
* NE -- Configure Kernel/MAC/Post
* TileDMADst -- Output DMA engine

It can work with 8 base addresses for the DMA streams per OP
* 2x Read, both used for things like sum
* 1x Write
* 1x T?
* 4x Kernel, though only the first one seems used

## Normal Flow for ANE Usage

* Keras/ONNX model -> coremltools
* CoreML model -> Espresso
* net.plist -> ANECompiler
* model.hwx -> ANEServices
* AppleH11ANEInterface, an IOKit interface to the kernel

## hwx file?

This is a Mach-O file. We haven't figured out all the details, but the ops are at 0x4000. See `hwx_parse.py`

