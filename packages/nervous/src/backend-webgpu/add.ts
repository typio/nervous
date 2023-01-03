import addWGSL from './add.wgsl?raw'

import { Tensor } from "../tensor"
import { gpuDevice } from '..'

export const add = async (a: Tensor, b: Tensor, axis?: number) => {
    if (b.constructor === Tensor) {

        if (JSON.stringify(a.shape()) !== JSON.stringify(b.shape()))
            throw new Error(`Currently tensors must be same shape, a.shape(): ${a.shape()} b.shape(): ${b.shape()}`)

        const aBufferSize = Math.max(32, a.data.byteLength)
        const bBufferSize = Math.max(32, b.data.byteLength)

        const aGPUBuffer = gpuDevice.createBuffer({
            mappedAtCreation: true,
            size: aBufferSize,
            usage: GPUBufferUsage.STORAGE
        })
        new Float32Array(aGPUBuffer.getMappedRange()).set(a.data)
        aGPUBuffer.unmap()

        const bGPUBuffer = gpuDevice.createBuffer({
            mappedAtCreation: true,
            size: bBufferSize,
            usage: GPUBufferUsage.STORAGE
        })
        new Float32Array(bGPUBuffer.getMappedRange()).set(b.data)
        bGPUBuffer.unmap()

        const resultGPUBuffer = gpuDevice.createBuffer({
            size: aBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })

        const readGPUBuffer = gpuDevice.createBuffer({
            size: aBufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })

        const computePipeline = gpuDevice.createComputePipeline({
            layout: "auto",
            compute: {
                module: gpuDevice.createShaderModule({
                    code: addWGSL
                }),
                entryPoint: "main"
            }
        })

        const bindGroup = gpuDevice.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: aGPUBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: bGPUBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: resultGPUBuffer
                    }
                }
            ]
        })

        const commandEncoder = gpuDevice.createCommandEncoder()
        const passEncoder = commandEncoder.beginComputePass()
        passEncoder.setPipeline(computePipeline)
        passEncoder.setBindGroup(0, bindGroup)
        passEncoder.dispatchWorkgroups(Math.ceil(a.flatValues().length / 64))
        passEncoder.end()

        commandEncoder.copyBufferToBuffer(resultGPUBuffer, 0, readGPUBuffer, 0, a.data.byteLength)
        gpuDevice.queue.submit([commandEncoder.finish()])
        await readGPUBuffer.mapAsync(GPUMapMode.READ)

        let result = new Float32Array(readGPUBuffer.getMappedRange().slice(0, a.data.byteLength))

        return new Tensor(result)
    } else {
        throw new Error("Adding a number is not yet supported in webGPU backend")
        // return webgpuExecuteTNT(a, b, addWGSL)
    }
}