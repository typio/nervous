import { gpuDevice } from '..'
import { Tensor } from '../tensor'

import {wgsl} from 'wgsl-preprocessor/wgsl-preprocessor.js'

/** create tensor of dot product */
export const dot = async (_a: Tensor, _m: Tensor) => {
  if (typeof _m === 'number') throw new Error('Please use Tensor.mul() for tensor scalar multiplication.')

  let a = _a
  let m = _m
  if (!_a.usingGPUBuffer) a = a.toGPU()

  if (!_m.usingGPUBuffer) m = m.toGPU()

  let aShape = a.webGPUBufferShape.map((e) => (e === 0 ? 1 : e)) // replacing 0s helps recognize a row vector
  let mShape = m.webGPUBufferShape.map((e) => (e === 0 ? 1 : e))
  let resShape = [0, 0, aShape[2], mShape[3]]
  let resSize = (4 + aShape[2] * mShape[3]) * Float32Array.BYTES_PER_ELEMENT

  if (aShape[3] !== mShape[2]) throw new Error('Tensors are not compatible shapes for multiplication.')

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
            // https://web.dev/gpu-compute/
            struct Matrix {
                s: vec4<f32>,
                v: array<f32>
            };

            @group(0) @binding(0) var<storage, read> aMatrix: Matrix;
            @group(0) @binding(1) var<storage, read> mMatrix:  Matrix;
            @group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= max(1, u32(aMatrix.s.z)) || global_id.y >= max(1, u32(mMatrix.s.w)) {
                    return;
                }

                outMatrix.s = vec4(0., 0., aMatrix.s.z, mMatrix.s.w);

                let outIndex: vec2<u32> = vec2(global_id.x, global_id.y);
                var sum: f32 = 0.0;
                for (var i: u32 = 0u; i < u32(aMatrix.s.w); i = i + 1u) {
                    let a: u32 = i + outIndex.x * u32(aMatrix.s.w);
                    let b: u32 = outIndex.y + i * u32(mMatrix.s.w);
                    sum += aMatrix.v[a] * mMatrix.v[b];
                }

                outMatrix.v[outIndex.y + outIndex.x * u32(mMatrix.s.w)] = sum;
            }
        `,
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
        resource: { buffer: m.webGPUBuffer },
      },
      {
        binding: 2,
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
