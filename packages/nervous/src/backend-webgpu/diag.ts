import { Tensor } from '../tensor'

import diagWGSL from './diag.wgsl?raw'

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

    const bindGroupLayout = gpuDevice.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'read-only-storage',
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'read-only-storage',
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage',
                },
            },
        ],
    })

    const computePipeline = gpuDevice.createComputePipeline({
        layout: gpuDevice.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        }),
        compute: {
            module: gpuDevice.createShaderModule({
                code: diagWGSL,
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
