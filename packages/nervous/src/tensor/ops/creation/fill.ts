import { Tensor } from '../tensor'
import { flatLengthFromShape, padShape } from '../tensorUtils'

import {wgsl} from 'wgsl-preprocessor/wgsl-preprocessor.js'

import { gpuDevice } from '..'

export const fill = (_shape: number | number[], value: number) => {
    // @ts-ignore
    let shape: number[] = padShape(_shape)
    let shapeArray = new Float32Array(shape)

    let valueFloat32 = new Float32Array([value])

    let resultSize = flatLengthFromShape(shape)

    const shapeGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: Math.max(32, shapeArray.byteLength),
        usage: GPUBufferUsage.STORAGE,
    })
    new Float32Array(shapeGPUBuffer.getMappedRange()).set(shapeArray)
    shapeGPUBuffer.unmap()

    const valueGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: Math.max(32, valueFloat32.byteLength),
        usage: GPUBufferUsage.STORAGE,
    })
    new Float32Array(valueGPUBuffer.getMappedRange()).set(valueFloat32)
    valueGPUBuffer.unmap()

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

@group(0) @binding(0) var<storage, read> shape: vec4<f32>;
@group(0) @binding(1) var<storage, read> value: f32;
@group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let thing = 2;
  outMatrix.shape = shape;
  outMatrix.values[global_id.x] = value;
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
                resource: { buffer: shapeGPUBuffer },
            },
            {
                binding: 1,
                resource: { buffer: valueGPUBuffer },
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
    passEncoder.dispatchWorkgroups(Math.ceil(resultSize / 64))
    passEncoder.end()

    gpuDevice.queue.submit([commandEncoder.finish()])
    return new Tensor(resultGPUBuffer, shape)
}
