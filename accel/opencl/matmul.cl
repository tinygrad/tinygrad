#define NUM_OUTPUTS 1

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

  float4 outputValues = (float4)(0, 0, 0, 0);

  for (short packedInputChannel = 0; packedInputChannel < numPackedInputChannelsForGroup; ++packedInputChannel) {
    inputLocation.x = packedInputChannel;
    float4 inputValues = read_imagef(input, smp, inputLocation);

    float4 weightValues[4];
    for (short outChIdx = 0; outChIdx < 4; ++outChIdx) {
      weightValues[outChIdx] = read_imagef(weights, smp, weightLocation);
      ++weightLocation.x;
    }

    outputValues.x += dot(inputValues, weightValues[0]);
    outputValues.y += dot(inputValues, weightValues[1]);
    outputValues.z += dot(inputValues, weightValues[2]);
    outputValues.w += dot(inputValues, weightValues[3]);
  }

  // insert unary and binary ops here
  int2 outputLocation;
  short outputColumn = startOutputColumn;
  outputLocation.y = outputRow;
  outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
  //BINOP
  ++outputColumn;

  // output to memory
  outputColumn = startOutputColumn;
  outputLocation.x = mad24(outputColumn, totalNumPackedOutputChannels, packedOutputChannel);
  write_imagef(output, outputLocation, outputValues);
  ++outputColumn;
}
