import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

import { Tensor } from '../tensor'
import { gpuDevice } from '..'
import { flatLengthFromShape, padShape } from '../tensorUtils'

export const gradientReLU = async (_a: Tensor, _b: Tensor) => {
  let a = _a
  let b = _b

  if (!a.usingGPUBuffer) a = await a.toGPU()
  if (!b.usingGPUBuffer) b = await b.toGPU()

  let aShape = padShape(a.webGPUBufferShape)
  let bShape = padShape(b.webGPUBufferShape)

  if (
    aShape[0] !== bShape[0] || //
    aShape[1] !== bShape[1] || //
    aShape[2] !== bShape[2] || //
    aShape[3] !== bShape[3]
  )
    throw new Error("tensor shapes don't match")

  let resSize = (4 + flatLengthFromShape(aShape)) * Float32Array.BYTES_PER_ELEMENT

  const resultGPUBuffer = gpuDevice.createBuffer({
    size: Math.max(32, resSize),
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
@group(0) @binding(1) var<storage, read> b:  Matrix;
@group(0) @binding(2) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    o.s = a.s;

    if global_id.x > u32(max(1., o.s.x) * max(1., o.s.y) * max(1., o.s.z) * max(1., o.s.w)) {
        return;
    }

    if (b.v[global_id.x] <= 0) {
        o.v[global_id.x] = 0.;
    } else {
        o.v[global_id.x] = a.v[global_id.x];
    }
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
        resource: {
          buffer: a.webGPUBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: b.webGPUBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: resultGPUBuffer,
        },
      },
    ],
  })

  const commandEncoder = gpuDevice.createCommandEncoder()
  const passEncoder = commandEncoder.beginComputePass()
  passEncoder.setPipeline(computePipeline)
  passEncoder.setBindGroup(0, bindGroup)
  passEncoder.dispatchWorkgroups(Math.ceil(flatLengthFromShape(aShape) / 64))
  passEncoder.end()

  gpuDevice.queue.submit([commandEncoder.finish()])
  return new Tensor(resultGPUBuffer, aShape)
}
