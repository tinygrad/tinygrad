#define NUM_OUTPUTS 4

//PREFIX

__kernel void image_conv(
  read_only image2d_t input,
  read_only image2d_t weights,
  write_only image2d_t output
#ifndef NOARGS
  ,short numPackedInputChannelsForGroup,
  short totalNumPackedInputChannels,
  short numPackedOutputChannelsForGroup,
  short totalNumPackedOutputChannels,
  short numOutputColumns,
  short numOutputRows, short numInputRows
#endif
  /*short filterSizeX, short filterSizeY,
  short paddingX, short paddingY,
  short strideX, short strideY,
  short dilationX, short dilationY*/
  //ARGS
  ) {

  //SHORTS

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  float4 outputValues[NUM_OUTPUTS];
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputValues[i] = (float4)(0, 0, 0, 0);
  }

  short packedOutputChannel = get_global_id(0);
  int2 weightLocation;
  weightLocation.x = 0;
  weightLocation.y = packedOutputChannel;

  short groupNum = (packedOutputChannel / numPackedOutputChannelsForGroup);
  short startPackedInputChannel = mul24(groupNum, numPackedInputChannelsForGroup);
  short startOutputColumn = mul24((short)get_global_id(1), NUM_OUTPUTS);
  short startX = mad24(mad24(startOutputColumn, strideX, -paddingX), totalNumPackedInputChannels, startPackedInputChannel);
  short strideWithChannels = mul24(strideX, totalNumPackedInputChannels);

  short outputRow = get_global_id(2);
  int2 inputLocation;

#ifdef BATCH
  // TODO: this doesn't work with y padding
  inputLocation.y = mad24(outputRow % numOutputRows, strideY, -paddingY);
  short batchOffset = (outputRow / numOutputRows) * numInputRows;
  inputLocation.y += batchOffset;
#else
  inputLocation.y = mad24(outputRow, strideY, -paddingY);
#endif

#ifdef DEPTHWISE_UNSTRIDED
  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    float4 inputValues[4];
    inputLocation.x = startX;
    for (short i = 1; i < 4; ++i) {
      inputValues[i] = read_imagef(input, smp, inputLocation);
      inputLocation.x += totalNumPackedOutputChannels;
    }
    for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {
      inputValues[0] = inputValues[1];
      inputValues[1] = inputValues[2];
      inputValues[2] = inputValues[3];
      inputValues[3] = read_imagef(input, smp, inputLocation);
      inputLocation.x += totalNumPackedInputChannels;
      float4 weightValues = read_imagef(weights, smp, weightLocation);
      ++weightLocation.x;
      outputValues[0] += inputValues[0] * weightValues;
      outputValues[1] += inputValues[1] * weightValues;
      outputValues[2] += inputValues[2] * weightValues;
      outputValues[3] += inputValues[3] * weightValues;
    }
    ++inputLocation.y;
  }
#else
  for (short rfRow = 0; rfRow < filterSizeY; ++rfRow) {
    // numPackedInputChannelsForGroup is 1 in depthwise
    for (short packedInputChannel = 0; packedInputChannel < numPackedInputChannelsForGroup; ++packedInputChannel) {
      short startXForChannel = startX + packedInputChannel;
      for (short rfColumn = 0; rfColumn < filterSizeX; ++rfColumn) {

        short dilatedStepX = mul24(totalNumPackedInputChannels, dilationX);
        inputLocation.x = mad24(rfColumn, dilatedStepX, startXForChannel);
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
    inputLocation.y += dilationY;
  }
#endif

  // insert unary and binary ops here
  int2 outputLocation;
  short outputColumn = startOutputColumn;
  outputLocation.y = outputRow;
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
    //BINOP
    ++outputColumn;
  }

  // output to memory
  outputColumn = startOutputColumn;
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
    if (outputColumn < numOutputColumns) {
      write_imagef(output, outputLocation, outputValues[i]);
    }
    ++outputColumn;
  }
}
