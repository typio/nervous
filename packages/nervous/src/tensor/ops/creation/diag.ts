import { Tensor } from '../tensor'

import {wgsl} from 'wgsl-preprocessor/wgsl-preprocessor.js'

import { gpuDevice } from '..'
import { shape } from './shape'
import { toArr } from '../tensorUtils'

export const diag = async (values: number[]) => {
    let valuesArray = new Float32Array(values)

    let resultSize = Math.pow(values.length, 2)

    const valuesGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: Math.max(32, valuesArray.byteLength),
        usage: GPUBufferUsage.STORAGE,
    })
    new Float32Array(valuesGPUBuffer.getMappedRange()).set(valuesArray)
    valuesGPUBuffer.unmap()

    let shapeArray = new Float32Array([0, 0, values.length, values.length])
    const shapeGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: Math.max(32, valuesArray.byteLength),
        usage: GPUBufferUsage.STORAGE,
    })
    new Float32Array(shapeGPUBuffer.getMappedRange()).set(shapeArray)
    shapeGPUBuffer.unmap()

    const resultBufferSize = Math.max(32, Float32Array.BYTES_PER_ELEMENT * (4 + resultSize))
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    const computePipeline = gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
            module: gpuDevice.createShaderModule({
                code: wgsl`
                struct Matrix {
    shape: vec4<f32>,
    values: array<f32>
};

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> shape: vec4<f32>;
@group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let side_length = u32(shape.z);
  if (global_id.x > side_length) {
    return;
  }

  outMatrix.shape = shape;

  // relies on the array buffer initially being filled with 0's
  outMatrix.values[global_id.x * (side_length + 1)] = values[global_id.x];
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
                resource: { buffer: valuesGPUBuffer },
            },
            {
                binding: 1,
                resource: { buffer: shapeGPUBuffer },
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
    passEncoder.dispatchWorkgroups(Math.ceil(values.length / 64))
    passEncoder.end()

    gpuDevice.queue.submit([commandEncoder.finish()])

    return new Tensor(resultGPUBuffer, toArr(shapeArray))
}
