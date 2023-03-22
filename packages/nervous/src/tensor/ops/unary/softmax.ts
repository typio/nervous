import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'
import { flatLengthFromShape } from '../tensorUtils'

import { gpuDevice } from '..'
import { Tensor } from '../tensor'

export const softmax = async (_a: Tensor, _dim?: number) => {
  let a = _a
  let dim = _dim === undefined ? 1 : _dim

  if (!a.usingGPUBuffer) a = await a.toGPU()

  const dimGPUBuffer = gpuDevice.createBuffer({
    mappedAtCreation: true,
    size: 32,
    usage: GPUBufferUsage.STORAGE,
  })
  new Uint32Array(dimGPUBuffer.getMappedRange()).set(new Uint32Array([dim]))
  dimGPUBuffer.unmap()

  let resSize = (4 + flatLengthFromShape(a.webGPUBufferShape)) * Float32Array.BYTES_PER_ELEMENT

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
@group(0) @binding(1) var<storage, read> dim:  u32;
@group(0) @binding(2) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= u32(a.s[3 - dim])) {
        return;
    }

    let row_n = u32(a.s[2]);
    let col_n = u32(a.s[3]);

    o.s = a.s;

    var sum :f32 = 0.0;

    var max = -0x1p-126f;
    if (dim == 1) {
        for (var i = 0u; i < col_n; i++) {
            let idx = global_id.x * col_n + i;
            let v = a.v[idx];
            if (v > max) {
                max = v;
            }
        }

        for (var i = 0u; i < col_n; i++) {
            let idx = global_id.x * col_n + i;
            sum += exp(a.v[idx] - max);
        }

        for (var i = 0u; i < col_n; i++) {
            let idx = global_id.x * col_n + i;
            o.v[idx] = exp(a.v[idx] - max) / sum;
        }
    } else if (dim == 0) {
        for (var i = 0u; i < row_n; i++) {
            let idx = global_id.x + col_n * i;
            let v = a.v[idx];
            if (v > max) {
                max = v;
            }
        }

        for (var i = 0u; i < row_n; i++) {
            let idx = u32(global_id.x + col_n * i);
            sum += exp(a.v[idx] - max);
        }

        for (var i = 0u; i < row_n; i++) {
            let idx = u32(global_id.x + col_n * i);
            o.v[idx] = exp(a.v[idx] - max) / sum;
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
          buffer: dimGPUBuffer,
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
  passEncoder.dispatchWorkgroups(Math.ceil(flatLengthFromShape(a.webGPUBufferShape) / 64))
  passEncoder.end()

  gpuDevice.queue.submit([commandEncoder.finish()])
  return new Tensor(resultGPUBuffer, a.webGPUBufferShape)
}
