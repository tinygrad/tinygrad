  #define NUM_OUTPUTS 4

  __kernel void conv(
    read_only image2d_t input,
    read_only image2d_t weights,
    write_only image2d_t output,
    short numPackedInputChannelsForGroup,
    short totalNumPackedInputChannels,
    short totalNumPackedOutputChannels,
    short numOutputColumns,
    short filterSizeX, short filterSizeY,
    short paddingX, short paddingY,
    short strideX, short strideY) {

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  float4 outputValues[NUM_OUTPUTS];
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputValues[i] = (float4)(0, 0, 0, 0);
  }

  short packedOutputChannel = get_global_id(0);
  int2 weightLocation;
  weightLocation.x = 0;
  weightLocation.y = packedOutputChannel;

  short startOutputColumn = mul24((short)get_global_id(1), NUM_OUTPUTS);
  short startX = mul24(mad24(startOutputColumn, strideX, -paddingX), totalNumPackedInputChannels);
  short strideWithChannels = mul24(strideX, totalNumPackedInputChannels);

  short outputRow = get_global_id(2);
  int2 inputLocation;
  inputLocation.y = mad24(outputRow, strideY, -paddingY); 

  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    // numPackedInputChannelsForGroup is 1 in depthwise
    for (short packedInputChannel = 0; packedInputChannel < numPackedInputChannelsForGroup; ++packedInputChannel) {
      short startXForChannel = startX + packedInputChannel;
      for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {

        inputLocation.x = mad24(rfColumn, totalNumPackedInputChannels, startXForChannel);
        float4 inputValues[NUM_OUTPUTS];
        for (short i = 0; i < NUM_OUTPUTS; ++i) {
          inputValues[i] = read_imagef(input, smp, inputLocation);
          inputLocation.x += strideWithChannels;
        }

#ifdef DEPTHWISE
        float4 weightValues = read_imagef(weights, smp, weightLocation);
        ++weightLocation.x;
        for (short i = 0; i < NUM_OUTPUTS; ++i) {
          outputValues[i] += inputValues[i] * weightValues;
        }
#else
        float4 weightValues[4];
        for (short outChIdx = 0; outChIdx < 4; ++outChIdx) {
          weightValues[outChIdx] = read_imagef(weights, smp, weightLocation);
          ++weightLocation.x;
        }

        for (short i = 0; i < NUM_OUTPUTS; ++i) {
          float4 curOutputValues = outputValues[i];
          curOutputValues.x += dot(inputValues[i], weightValues[0]);
          curOutputValues.y += dot(inputValues[i], weightValues[1]);
          curOutputValues.z += dot(inputValues[i], weightValues[2]);
          curOutputValues.w += dot(inputValues[i], weightValues[3]);
          outputValues[i] = curOutputValues;
        }
#endif
      }
    }
    ++inputLocation.y;
  }

  // insert unary and binary ops here

  // output to memory
  int2 outputLocation;
  short outputColumn = startOutputColumn;
  outputLocation.y = outputRow;
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
    if (outputColumn < numOutputColumns) {
      write_imagef(output, outputLocation, outputValues[i]);
    }
    ++outputColumn;
  }
}
