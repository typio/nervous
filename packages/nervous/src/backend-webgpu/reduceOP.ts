import reduceWGSL from './reduceOP.wgsl?raw'
import { flatLengthFromShape } from '../tensorUtils'

import { gpuDevice } from '..'
import { ReduceOP, Tensor } from '../tensor'

/** returns scalar sum in Tensor of all tensor values, in case of 2d matrix, axis can be specified for vector of sums: 0 for columns 1 for rows */
export const reduceOP = async (_a: Tensor, flag: ReduceOP, _axis?: 0 | 1): Promise<Tensor> => {
    let a = _a

    if (!a.usingGPUBuffer) a = await a.toGPU()

    // has value -1 if no axis is given
    let axis = _axis === undefined ? -1 : _axis

    const axisGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: 32,
        usage: GPUBufferUsage.STORAGE,
    })
    new Int32Array(axisGPUBuffer.getMappedRange()).set(new Int32Array([axis]))
    axisGPUBuffer.unmap()

    let resSize: number

    if (axis === 0) resSize = (4 + a.webGPUBufferShape[2]) * Float32Array.BYTES_PER_ELEMENT
    else if (axis === 1) resSize = (4 + a.webGPUBufferShape[3]) * Float32Array.BYTES_PER_ELEMENT
    else resSize = (4 + 1) * Float32Array.BYTES_PER_ELEMENT

    const flagGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: 32,
        usage: GPUBufferUsage.STORAGE,
    })
    new Uint32Array(flagGPUBuffer.getMappedRange()).set(new Uint32Array([flag]))
    flagGPUBuffer.unmap()

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
                code: reduceWGSL,
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
                    buffer: axisGPUBuffer,
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
    passEncoder.dispatchWorkgroups(Math.ceil(flatLengthFromShape(a.webGPUBufferShape) / 64))
    passEncoder.end()

    gpuDevice.queue.submit([commandEncoder.finish()])
    return new Tensor(resultGPUBuffer, a.webGPUBufferShape)
}
