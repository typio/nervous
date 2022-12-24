import { Tensor } from "../tensor";
import { gpuDevice } from "..";

import type { Rank1To4Array, BinaryOp } from "../tensor";
// import "@webgpu/types"

import addWGSL from "./bop_add.wgsl?raw"
import subWGSL from "./bop_sub.wgsl?raw"
import mulWGSL from "./bop_mul.wgsl?raw"
import divWGSL from "./bop_div.wgsl?raw"
import modWGSL from "./bop_mod.wgsl?raw"

import randomNormalWGSL from "./gen_randomNormal.wgsl?raw"


import { flatLengthFromShape } from "../tensorUtils";

// tensor op tensor -> tensor
const webgpuExecuteTTT = async (a: Tensor, b: Tensor, code, axis?, flags?: {}): Promise<Tensor> => {
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

// tensor op number -> tensor
const webgpuExecuteTNT = (a: Tensor, b: BinaryOp) => {

}

// f(shape) -> tensor  | e.g. random
const webgpuExecuteST = async (shape: number[], code, flags?: {}) => {
    let shapeArray = new Int8Array(shape)
    const shapeGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: shapeArray.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    new Float32Array(shapeGPUBuffer.getMappedRange()).set(shapeArray)
    shapeGPUBuffer.unmap()

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

    return tensor(new Float32Array(addReadGPUBuffer.getMappedRange()))
}

// f(tensor) -> tensor | e.g. sum
const webgpuExecuteT = () => {

}

const scalar = (value: number) => {
    return new Tensor(value)
}

const tensor = (values: number | Rank1To4Array, shape?: number[]) => {
    if (values.constructor === Array && values.length === 1) return new Tensor(values[0])
    return new Tensor(values, shape)
}

const randomNormal = (shape: number[], mean?: number, std?: number) => {

}

const add = async (a: Tensor, b: Tensor) => {
    if (a.rank !== 2 || b.rank !== 2) {
        throw new Error("addTensor input be 2d arrays")
    }
    return webgpuExecuteTTT(a, b, addWGSL)
}

export default { scalar, tensor, randomNormal, add }