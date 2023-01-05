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


    // if (aRank > 2)
    //     throw new Error("Tensor matmul on rank > 2 tensors not yet supported.")

    // const aValuesGPUBuffer = gpuDevice.createBuffer({
    //     mappedAtCreation: true,
    //     size: Math.max(32, a.data.byteLength),
    //     usage: GPUBufferUsage.STORAGE
    // })
    // new Float32Array(aValuesGPUBuffer.getMappedRange()).set(a.data)
    // aValuesGPUBuffer.unmap()

    // const mValuesGPUBuffer = gpuDevice.createBuffer({
    //     mappedAtCreation: true,
    //     size: Math.max(32, m.data.byteLength),
    //     usage: GPUBufferUsage.STORAGE
    // })
    // new Float32Array(mValuesGPUBuffer.getMappedRange()).set(m.data)
    // mValuesGPUBuffer.unmap()

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
    passEncoder.dispatchWorkgroups(Math.ceil(aShape.at(0) / 8), Math.ceil(mShape.at(1) / 8))

    passEncoder.end()
    gpuDevice.queue.submit([commandEncoder.finish()])

    return new Tensor(resultGPUBuffer, resShape)

    // commandEncoder.copyBufferToBuffer(resultGPUBuffer, 0, readGPUBuffer, 0, Float32Array.BYTES_PER_ELEMENT * (4 + aShape.at(0) * mShape.at(-1)))
    // gpuDevice.queue.submit([commandEncoder.finish()])
    // await readGPUBuffer.mapAsync(GPUMapMode.READ)

    // let result = new Float32Array(readGPUBuffer.getMappedRange())
    // console.log(result);

    // return new Tensor(result)
}   