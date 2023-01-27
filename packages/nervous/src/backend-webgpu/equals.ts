// TODO: Remove this, think on better solutions for comparing outputs and expected outputs

import { gpuDevice } from '..'
import { Tensor } from '../tensor'
import { flatLengthFromShape } from '../tensorUtils'

import equalsWGSL from './equals.wgsl?raw'

export const equals = async (_a: Tensor, _m: Tensor): Promise<Boolean> => {
    let a = _a
    let m = _m
    if (!_a.usingGPUBuffer) {
        a = await a.toGPU()
    }

    if (!_m.usingGPUBuffer) {
        m = await m.toGPU()
    }

    let aLength = flatLengthFromShape(a.webGPUBufferShape)

    if (aLength !== flatLengthFromShape(m.webGPUBufferShape)) return false

    let resSize = Uint32Array.BYTES_PER_ELEMENT

    const resultBufferSize = Math.max(4, resSize)
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(resultBufferSize, 4),
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
                code: equalsWGSL,
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

    const readGPUBuffer = gpuDevice.createBuffer({
        size: resultGPUBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })

    const commandEncoder = gpuDevice.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(Math.ceil(aLength / 64))

    passEncoder.end()

    commandEncoder.copyBufferToBuffer(resultGPUBuffer, 0, readGPUBuffer, 0, 4)
    gpuDevice.queue.submit([commandEncoder.finish()])
    await readGPUBuffer.mapAsync(GPUMapMode.READ)

    let result = new Uint32Array(readGPUBuffer.getMappedRange())
    return result[0] === 1
}
