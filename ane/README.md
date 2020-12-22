# The Apple Neural Engine

The Apple Neural Engine is a fancy DMA Engine that is based around convolutions. We don't have all the details worked out yet, but we can do some things with it. At its core, it runs through 0x300 ops in an hwx file. See `aneregs.json` for the registers used in each op.

It works with 5D Tensors, you specify the stride for the latter 4. All strides must be a multiple of 0x40 bytes
* Column
* Row
* Plane (height/channels)
* Depth
* Group (batch)

It works with 3 data types
* UInt8
* Int8
* Float16

The ops have several parts
* Header -- The base addresses for the DMA engines
* KernelDMASrc -- 16x wide DMA engine for the weights/bias/nonlinearity
* Common -- Specifies the parameters for the convolution
* TileDMASrc -- Input DMA engine
* L2 -- L2 caching for Source/Result?
* NE -- Configure Kernel/MAC/Post
* TileDMADst -- Output DMA engine

It can work with 8 base addresses for the DMA streams per OP
* 2x Read
* 1x Write
* 1x T?
* 4x Kernel, I assume each is the base for 4 of the engines

## Normal Flow for ANE Usage

* Keras/ONNX model -> coremltools
* CoreML model -> Espresso
* net.plist -> ANECompiler
* model.hwx -> ANEServices
* AppleH11ANEInterface, an IOKit interface to the kernel

## hwx file?

This is a Mach-O file. We haven't figured out all the details, but the ops are at 0x4000. See `hwx_parse.py`

