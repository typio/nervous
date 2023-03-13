import { gpuDevice } from '..'
import { Tensor } from '../tensor'
import { flatLengthFromShape } from '../tensorUtils'

import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

/** switch rows and columns of a >=2d Tensor */
export const transpose = async (_a: Tensor) => {
  let a = _a
  if (!_a.usingGPUBuffer) {
    a = a.toGPU()
  }

  // its a scalar so we are done
  if (flatLengthFromShape(a.webGPUBufferShape) === 1) {
    return _a
  }

  let resShape = [0, 0, a.webGPUBufferShape[3], a.webGPUBufferShape[2]]
  let resSize = (4 + flatLengthFromShape(a.webGPUBufferShape)) * Float32Array.BYTES_PER_ELEMENT

  const resultBufferSize = Math.max(32, resSize)
  const resultGPUBuffer = gpuDevice.createBuffer({
    size: Math.max(resultBufferSize, 32),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const computePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: {
      module: gpuDevice.createShaderModule({
        code: wgsl`
        struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= max(1, u32(a.s.z)) || global_id.y >= max(1, u32(a.s.w)) {
        return;
    }

    // TODO: try to speed up with shared memory
    o.s = vec4(0, 0, a.s[3], a.s[2]);
    o.v[global_id.x + u32(a.s[2]) * global_id.y] = a.v[global_id.x * u32(a.s[3]) + global_id.y];
}
`
      }),
      entryPoint: 'main',
    },
  })

  const bindGroup = gpuDevice.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: a.webGPUBuffer },
      },
      {
        binding: 1,
        resource: { buffer: resultGPUBuffer },
      },
    ],
  })

  const commandEncoder = gpuDevice.createCommandEncoder()
  const passEncoder = commandEncoder.beginComputePass()
  passEncoder.setPipeline(computePipeline)
  passEncoder.setBindGroup(0, bindGroup)
  passEncoder.dispatchWorkgroups(Math.ceil(resShape[2] / 8), Math.ceil(resShape[3] / 8))

  passEncoder.end()
  gpuDevice.queue.submit([commandEncoder.finish()])

  return new Tensor(resultGPUBuffer, resShape)
}
