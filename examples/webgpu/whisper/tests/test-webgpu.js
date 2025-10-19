import fs from 'node:fs';
import { PNG } from 'pngjs';
import { create, globals } from 'webgpu';

(async () => {
//const { create, globals } = require('webgpu');
//import { create, globals } from 'webgpu';
//const { create, globals } = await import('webgpu');

//Object.assign(globalThis, globals);
//const navigator = { gpu: create([]) };

//const fs = require('node:fs');
//const { PNG } = require('pngjs');
//const { create, globals } = require('webgpu');

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };

const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();

const module = device.createShaderModule({
  code: `
    @vertex fn vs(
      @builtin(vertex_index) vertexIndex : u32
    ) -> @builtin(position) vec4f {
      let pos = array(
        vec2f( 0.0,  0.5),
        vec2f(-0.5, -0.5),
        vec2f( 0.5, -0.5),
      );

      return vec4f(pos[vertexIndex], 0.0, 1.0);
    }

    @fragment fn fs() -> @location(0) vec4f {
      return vec4f(1, 0, 0, 1);
    }
  `,
});

const pipeline = await device.createRenderPipelineAsync({
  layout: 'auto',
  vertex: { module },
  fragment: { module, targets: [{ format: 'rgba8unorm' }] },
});

const texture = device.createTexture({
  format: 'rgba8unorm',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  size: [256, 256],
});

const align = (v, alignment) => Math.ceil(v / alignment) * alignment;

const bytesPerRow = align(texture.width * 4, 256);
const buffer = device.createBuffer({
  size: bytesPerRow * texture.height,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
  colorAttachments: [
    {
      view: texture.createView(),
      clearValue: [0.3, 0.3, 0.3, 1],
      loadOp: 'clear',
      storeOp: 'store',
    },
  ],
});
pass.setPipeline(pipeline);
pass.draw(3);
pass.end();
encoder.copyTextureToBuffer(
  { texture },
  { buffer, bytesPerRow },
  [ texture.width, texture.height ],
);
const commandBuffer = encoder.finish();
device.queue.submit([commandBuffer]);

await buffer.mapAsync(GPUMapMode.READ);
const rawPixelData = buffer.getMappedRange();

const png = new PNG({
  width: texture.width,
  height: texture.height,
  filterType: -1,
});

const dstBytesPerRow = texture.width * 4;
for (let y = 0; y < texture.height; y++) {
  const dst = (texture.width * y) * 4;
  const src = (y * bytesPerRow);
  png.data.set(new Uint8Array(rawPixelData, src, dstBytesPerRow), dst);
}

// Write the PNG to a file
fs.writeFileSync('output.png', PNG.sync.write(png, {colorType: 6}));
console.log('PNG file has been saved as output.png');
}) ();