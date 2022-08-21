//PREFIX

__kernel void matmul(
  write_only image2d_t output,
  __local float *outputScratch,
  read_only image1d_t input,
  read_only image2d_t weights
  //ARGS
  ) {

  //SHORTS

  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  short packedOutputChannel = get_global_id(2);
  short scratchOffset = mad24((short)get_local_id(1), 4, (short)get_local_id(0));
  short weightIndex = (short)get_global_id(0);

  // fast path precompute (32x speedup)
  float outputValue = 0.0f;
  for (short inputSet = (short)get_global_id(1); inputSet < numPackedInputChannelsForGroup; inputSet += get_global_size(1)) {
    float4 inputValues = read_imagef(input, smp, inputSet);
    float4 weightValues = read_imagef(weights, smp, (int2)(mad24(inputSet, 4, weightIndex), packedOutputChannel));
    outputValue += dot(inputValues, weightValues);
  }

  short scratchIndex = mad24((short)get_local_id(2), mul24((short)get_local_size(1), 4), scratchOffset);
  outputScratch[scratchIndex] = outputValue;
  //barrier(CLK_LOCAL_MEM_FENCE);

  if (scratchOffset == 0) {
    float4 outputValues = (float4)(0, 0, 0, 0);

    // fast path
    for (short i = 0; i < (short)get_global_size(1); ++i) {
      outputValues += vload4(0, &outputScratch[scratchIndex]);
      scratchIndex += 4;
    }

    // slow path
    /*for (short packedInputChannel = 0; packedInputChannel < numPackedInputChannelsForGroup; ++packedInputChannel) {
      float4 inputValues = read_imagef(input, smp, packedInputChannel);

      float4 weightValues[4];
      for (short outChIdx = 0; outChIdx < 4; ++outChIdx) {
        weightValues[outChIdx] = read_imagef(weights, smp, (int2)(packedInputChannel*4 + outChIdx, packedOutputChannel));
      }

      outputValues.x += dot(inputValues, weightValues[0]);
      outputValues.y += dot(inputValues, weightValues[1]);
      outputValues.z += dot(inputValues, weightValues[2]);
      outputValues.w += dot(inputValues, weightValues[3]);
    }*/

    // insert unary and binary ops here
    int2 outputLocation;
    outputLocation.x = packedOutputChannel;
    //BINOP

    // output to memory
    write_imagef(output, outputLocation, outputValues);
  }
}
