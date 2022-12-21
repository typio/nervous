import * as nv from "nervous"
import "@webgpu/types"

import addWGSL from "./add.wgsl"

let device: null | GPUDevice = null;

const initDevice = async () => {
    if (!('gpu' in navigator)) {
        console.error("User agent doesn't support WebGPU.");
        return false;
    }

    const gpuAdapter = await navigator.gpu.requestAdapter();

    if (!gpuAdapter) {
        console.error('No WebGPU adapters found.');
        return false;
    } else {
        // console.log(gpuAdapter)
    }

    device = await gpuAdapter.requestDevice();

    device.lost.then((info) => {
        console.error(`WebGPU device was lost: ${info.message}`);
        device = null;
        if (info.reason != 'destroyed') {
            initDevice();
        }
    });


    console.log(device)

    let a = nv.randomNormal([128, 128])
    let b = nv.randomNormal([128, 128])
    console.log(a, b);

    addTensor(device, a, b);

    return true;
}

const addTensor = async (device: GPUDevice, a: nv.Tensor, b: nv.Tensor) => {
    if (a.rank !== 2 || b.rank !== 2) {
        throw new Error("addTensor input be 2d arrays")
    }
    const aShapeArray = new Int32Array([a.shape[0], a.shape[1]])
    const aShapeGPUBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: aShapeArray.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    new Int32Array(aShapeGPUBuffer.getMappedRange()).set(aShapeArray)
    aShapeGPUBuffer.unmap()

    const bShapeArray = new Int32Array([b.shape[0], b.shape[1]])
    const bShapeGPUBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: bShapeArray.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    new Int32Array(bShapeGPUBuffer.getMappedRange()).set(bShapeArray)
    bShapeGPUBuffer.unmap()

    const aValuesArray = new Float32Array(a.values)
    const aValuesGPUBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: aValuesArray.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    new Float32Array(aValuesGPUBuffer.getMappedRange()).set(aValuesArray)
    aValuesGPUBuffer.unmap()

    const bValuesArray = new Float32Array(b.values)
    const bValuesGPUBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: bValuesArray.byteLength,
        usage: GPUBufferUsage.STORAGE
    })
    new Float32Array(bValuesGPUBuffer.getMappedRange()).set(bValuesArray)
    bValuesGPUBuffer.unmap()


    const addResultGPUBuffer = device.createBuffer({
        size: aShapeArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    })

    const addReadGPUBuffer = device.createBuffer({
        size: aShapeArray.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const bindGroupLayout = device.createBindGroupLayout({
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
                    type: "read-only-storage"
                }
            } as GPUBindGroupLayoutEntry,
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage"
                }
            } as GPUBindGroupLayoutEntry,
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage"
                }
            } as GPUBindGroupLayoutEntry
        ]
    })

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: aShapeGPUBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: bShapeGPUBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: aValuesGPUBuffer
                }
            },
            {
                binding: 3,
                resource: {
                    buffer: bValuesGPUBuffer
                }
            },
            {
                binding: 4,
                resource: {
                    buffer: addResultGPUBuffer
                }
            }
        ]
    })

    const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
            module: device.createShaderModule({
                code: addWGSL
            }),
            entryPoint: "main"
        }
    })
}

initDevice();