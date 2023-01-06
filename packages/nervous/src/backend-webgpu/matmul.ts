import { gpuDevice } from ".."
import { Tensor } from "../tensor"

import matmulWGSL from "./matmul.wgsl?raw"

/** create tensor of dot product */
export const matmul = async (_a: Tensor, _m: Tensor) => {
    if (typeof _m === 'number')
        throw new Error("Please use Tensor.mul() for tensor scalar multiplication.")

    let a = _a
    let m = _m
    if (!_a.usingGPUBuffer) {
        a = await a.toGPU()
    }

    if (!_m.usingGPUBuffer) {
        m = await m.toGPU()
    }

    let aShape = a.webGPUBufferShape
    let mShape = m.webGPUBufferShape
    let resShape = [a.webGPUBufferShape[0], m.webGPUBufferShape[1]]
    let resSize = (4 + a.webGPUBufferShape[0] * m.webGPUBufferShape[1]) * Float32Array.BYTES_PER_ELEMENT

    if (aShape.at(-1) !== mShape.at(0))
        throw new Error("Tensors are not compatible shapes for multiplication.")

    const resultBufferSize = Math.max(32, resSize)
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
                resource: { buffer: a.webGPUBuffer }
            },
            {
                binding: 1,
                resource: { buffer: m.webGPUBuffer }
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
    passEncoder.dispatchWorkgroups(Math.ceil(resShape[0] / 8), Math.ceil(resShape[1] / 8))

    passEncoder.end()
    gpuDevice.queue.submit([commandEncoder.finish()])

    return new Tensor(resultGPUBuffer, resShape)
}   