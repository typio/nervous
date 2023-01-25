import { gpuDevice } from '..'
import { Tensor } from '../tensor'
import { flatLengthFromShape, padShape } from '../tensorUtils'

import repeatWGSL from './repeat.wgsl?raw'

export const repeat = async (_a: Tensor, _scales: number[]): Promise<Tensor> => {
    let a = _a
    if (!a.usingGPUBuffer) a = await a.toGPU()

    let scales = padShape(_scales)
    let scalesFloatArray = new Float32Array(scales)
    const scalesGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: scalesFloatArray.byteLength,
        usage: GPUBufferUsage.STORAGE,
    })
    new Float32Array(scalesGPUBuffer.getMappedRange()).set(scalesFloatArray)
    scalesGPUBuffer.unmap()

    let resShape = a.webGPUBufferShape.map((e, i) => e * scales[i])
    console.log(resShape)

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
                    type: 'storage',
                },
            },
        ],
    })

    const computePipeline = gpuDevice.createComputePipeline({
        // layout: 'auto',
        layout: gpuDevice.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        }),
        compute: {
            module: gpuDevice.createShaderModule({
                code: repeatWGSL,
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
                    buffer: scalesGPUBuffer,
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
    passEncoder.dispatchWorkgroups(Math.ceil(flatLengthFromShape(resShape) / 64))
    passEncoder.end()

    gpuDevice.queue.submit([commandEncoder.finish()])
    return new Tensor(resultGPUBuffer, resShape)
}
