import addWGSL from './add.wgsl?raw'

import { Tensor } from "../tensor"
import { gpuDevice } from '..'

export const add = async (a: Tensor, b: Tensor, axis?: number) => {
    if (b.constructor === Tensor) {
        if (a.rank !== 2 || b.rank !== 2) 
            throw new Error("Tensors must be 2d")
        
        const aValuesGPUBuffer = gpuDevice.createBuffer({
            mappedAtCreation: true,
            size: a.values.byteLength,
            usage: GPUBufferUsage.STORAGE
        })
        new Float32Array(aValuesGPUBuffer.getMappedRange()).set(a.values)
        aValuesGPUBuffer.unmap()

        const bValuesArray = new Float32Array(b.values)
        const bValuesGPUBuffer = gpuDevice.createBuffer({
            mappedAtCreation: true,
            size: bValuesArray.byteLength,
            usage: GPUBufferUsage.STORAGE
        })
        new Float32Array(bValuesGPUBuffer.getMappedRange()).set(bValuesArray)
        bValuesGPUBuffer.unmap()

        const resultGPUBuffer = gpuDevice.createBuffer({
            size: a.values.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        })

        const readGPUBuffer = gpuDevice.createBuffer({
            size: a.values.byteLength,
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
                        buffer: aValuesGPUBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: bValuesGPUBuffer
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
        passEncoder.dispatchWorkgroups(a.values.length)
        passEncoder.end()

        commandEncoder.copyBufferToBuffer(resultGPUBuffer, 0, readGPUBuffer, 0, a.values.byteLength)
        gpuDevice.queue.submit([commandEncoder.finish()])
        await readGPUBuffer.mapAsync(GPUMapMode.READ)

        let result = new Float32Array(readGPUBuffer.getMappedRange())
        return new Tensor(result, a.shape)
    } else {
        throw new Error("Adding a number is not yet supported in webGPU backend")
        // return webgpuExecuteTNT(a, b, addWGSL)
    }
}