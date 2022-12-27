import { gpuDevice } from ".."
import { Tensor } from "../tensor"
import { tensor } from "./tensor"

/// <reference types="@webgpu/types" />

// tensor op tensor -> tensor
export const webgpuExecuteTTT = async (a: Tensor, b: Tensor, code, axis?, flags?: {}): Promise<Tensor> => {
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

    const addResultGPUBuffer = gpuDevice.createBuffer({
        size: a.values.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    })

    const addReadGPUBuffer = gpuDevice.createBuffer({
        size: a.values.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const bindGroupLayout = gpuDevice.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            } as GPUBindGroupLayoutEntry,
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            } as GPUBindGroupLayoutEntry,
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            } as GPUBindGroupLayoutEntry
        ]
    })

    const bindGroup = gpuDevice.createBindGroup({
        layout: bindGroupLayout,
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
                    buffer: addResultGPUBuffer
                }
            }
        ]
    })

    const computePipeline = gpuDevice.createComputePipeline({
        layout: gpuDevice.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: gpuDevice.createShaderModule({
                code
            }),
            entryPoint: "main"
        }
    })

    const commandEncoder = gpuDevice.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(a.values.length)
    passEncoder.end()

    commandEncoder.copyBufferToBuffer(addResultGPUBuffer, 0, addReadGPUBuffer, 0, a.values.byteLength)
    gpuDevice.queue.submit([commandEncoder.finish()])
    await addReadGPUBuffer.mapAsync(GPUMapMode.READ)

    let result = new Float32Array(addReadGPUBuffer.getMappedRange())
    return tensor(result, a.shape)
}