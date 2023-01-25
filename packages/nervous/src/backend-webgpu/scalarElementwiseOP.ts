import scalarElementwiseOPWGSL from './scalarElementwiseOP.wgsl?raw'

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
                    type: 'read-only-storage',
                },
            },
            {
                binding: 3,
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
                code: scalarElementwiseOPWGSL,
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
