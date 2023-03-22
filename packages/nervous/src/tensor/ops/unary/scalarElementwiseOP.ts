import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

import { ScalarElementwiseOP, Tensor } from '../tensor'
import { gpuDevice } from '..'
import { flatLengthFromShape, padShape } from '../tensorUtils'

export const scalarElementwiseOP = async (_a: Tensor, n: number, flag: ScalarElementwiseOP) => {
  let a = _a

  if (!a.usingGPUBuffer) a = await a.toGPU()

  let aShape = padShape(a.webGPUBufferShape)

  const scalarGPUBuffer = gpuDevice.createBuffer({
    mappedAtCreation: true,
    size: 32,
    usage: GPUBufferUsage.STORAGE,
  })
  new Float32Array(scalarGPUBuffer.getMappedRange()).set(new Float32Array([n]))
  scalarGPUBuffer.unmap()

  const flagGPUBuffer = gpuDevice.createBuffer({
    mappedAtCreation: true,
    size: 32,
    usage: GPUBufferUsage.STORAGE,
  })
  new Uint32Array(flagGPUBuffer.getMappedRange()).set(new Uint32Array([flag]))
  flagGPUBuffer.unmap()

  let resShape = aShape

  let resSize = (4 + flatLengthFromShape(resShape)) * Float32Array.BYTES_PER_ELEMENT

  const resultGPUBuffer = gpuDevice.createBuffer({
    size: Math.max(32, resSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const computePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: {
      module: gpuDevice.createShaderModule({
        code: wgsl`struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> n:  f32;
@group(0) @binding(2) var<storage, read> flag: u32;
@group(0) @binding(3) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  o.s = a.s;

  if global_id.x > u32(max(1., o.s.x) * max(1., o.s.y) * max(1., o.s.z) * max(1., o.s.w)) {
      return;
  }

  let i = global_id.x;

  switch flag {
      case 0u: { // log
          o.v[i] = log(a.v[i]) / log(n);
      }
      case 1u: { // pow
        o.v[i] = pow(a.v[i], n);
      }
      case 2u: { // applyMax
        o.v[i] = max(a.v[i], n);
      }
      case 3u: { // applyMax
        o.v[i] = min(a.v[i], n);
      }
      case 4u: { // exp
        o.v[i] = pow(n, a.v[i]);
      }
      default: {
          o.v[i] = a.v[i];
      }
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
          buffer: scalarGPUBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: flagGPUBuffer,
        },
      },
      {
        binding: 3,
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
  passEncoder.dispatchWorkgroups(Math.ceil(flatLengthFromShape(resShape) / 64))
  passEncoder.end()

  gpuDevice.queue.submit([commandEncoder.finish()])
  return new Tensor(resultGPUBuffer, resShape)
}
