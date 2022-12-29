import { gpuDevice } from ".."
import { Tensor } from "../tensor"

import matmulWGSL from "./matmul.wgsl?raw"

/** create tensor of dot product */
export const matmul = async (a: Tensor, m: Tensor) => {
    let aRank = a.rank()
    let aShape = a.shape()
    let mShape = m.shape()

    if (typeof m === 'number' || aRank === 0) {
        throw new Error("Please use Tensor.mul() for tensor scalar multiplication.")
    }

    if (aShape.at(-1) !== mShape.at(0))
        throw new Error("Tensors are not compatible shapes for multiplication.")

    if (aRank > 2)
        throw new Error("Tensor matmul on rank > 2 tensors not yet supported.")


    const aValuesGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: a.data.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    new Float32Array(aValuesGPUBuffer.getMappedRange()).set(a.data)
    aValuesGPUBuffer.unmap()

    // const mValuesArray = new Float32Array(m.data) // remove this and see performance diff
    const mValuesGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: m.data.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    new Float32Array(mValuesGPUBuffer.getMappedRange()).set(m.data)
    mValuesGPUBuffer.unmap()

    const resultBufferSize = Float32Array.BYTES_PER_ELEMENT * (4 + aShape.at(0) * mShape.at(-1))
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    })

    const readGPUBuffer = gpuDevice.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const computePipeline = gpuDevice.createComputePipeline({
        layout: "auto",
        compute: {
            module: gpuDevice.createShaderModule({
                code: matmulWGSL
            }),
            entryPoint: "main"
        }
    })

    const bindGroup = gpuDevice.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: aValuesGPUBuffer }
            },
            {
                binding: 1,
                resource: { buffer: mValuesGPUBuffer }
            },
            {
                binding: 2,
                resource: { buffer: resultGPUBuffer }
            }
        ]
    })

    const commandEncoder = gpuDevice.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(Math.ceil(aShape.at(-1) / 8), Math.ceil(mShape.at(0) / 8))
    passEncoder.end()

    commandEncoder.copyBufferToBuffer(resultGPUBuffer, 0, readGPUBuffer, 0, resultBufferSize)
    gpuDevice.queue.submit([commandEncoder.finish()])
    await readGPUBuffer.mapAsync(GPUMapMode.READ)

    let result = new Float32Array(readGPUBuffer.getMappedRange())

    return new Tensor(result)
}   