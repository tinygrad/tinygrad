#define NUM_OUTPUTS 4

//PREFIX

__kernel void matmul(
  read_only image2d_t input,
  read_only image2d_t weights,
  write_only image2d_t output
  //ARGS
  ) {

  //SHORTS

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  short packedOutputChannel = get_global_id(0);
  int2 weightLocation;
  weightLocation.x = 0;
  weightLocation.y = packedOutputChannel;

  short startOutputColumn = mul24((short)get_global_id(1), NUM_OUTPUTS);

  short outputRow = get_global_id(2);
  int2 inputLocation;
  inputLocation.y = outputRow;

  float4 outputValues[NUM_OUTPUTS];
  for (short i = 0; i < NUM_OUTPUTS; ++i) {
    outputValues[i] = (float4)(0, 0, 0, 0);
  }

  for (short packedInputChannel = 0; packedInputChannel < numPackedInputChannelsForGroup; ++packedInputChannel) {
    inputLocation.x = packedInputChannel;
    
    float4 inputValues[NUM_OUTPUTS];
    for (short i = 0; i < NUM_OUTPUTS; ++i) {
      inputValues[i] = read_imagef(input, smp, inputLocation);
      inputLocation.x += totalNumPackedInputChannels;
    }

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
  }

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
