// import nv from "nervous"

// // console.log(nv);
// const main = async () => {
//     let start = performance.now()
//     await nv.init({ backend: 'webgpu' })
//     console.log(await nv.add(nv.randomNormal([1,10]), new nv.Tensor([[9, 9],[5,6]])))
//     // console.log(await nv.add(new nv.Tensor([[1, 2],[3,4]]), new nv.Tensor([[9, 9],[5,6]])))

// }

// main()


// // import "@webgpu/types"

// // import addWGSL from "./add.wgsl"

// // let device: null | GPUDevice = null;

// // const initDevice = async () => {
// //     if (!('gpu' in navigator)) {
// //         console.error("User agent doesn't support WebGPU.");
// //         return false;
// //     }

// //     const gpuAdapter = await navigator.gpu.requestAdapter();

// //     if (!gpuAdapter) {
// //         console.error('No WebGPU adapters found.');
// //         return false;
// //     } else {
// //         // console.log(gpuAdapter)
// //     }

// //     device = await gpuAdapter.requestDevice();

// //     device.lost.then((info) => {
// //         console.error(`WebGPU device was lost: ${info.message}`);
// //         device = null;
// //         if (info.reason != 'destroyed') {
// //             initDevice();
// //         }
// //     });


// //     console.log(device)

// //     let size = 512
// //     let a = nv.randomNormal([size, size])
// //     let b = nv.randomNormal([size, size])

// //     var startTime = performance.now()

// //     let webgpuadd = nv.tensor(await addTensor(device, a, b), a.shape)
// //     console.log('sup');
    
// //     var midTime = performance.now()
// //     let jsadd = a.add(b)
// //     var endTime = performance.now()

// //     console.log(`WebGPU took ${midTime - startTime}. JS took ${endTime - midTime}.`)

// //     console.log(webgpuadd.sum().getValues(), jsadd.sum().getValues());

// //     return true;
// // }

// // const addTensor = async (device: GPUDevice, a: nv.Tensor, b: nv.Tensor) => {
// //     if (a.rank !== 2 || b.rank !== 2) {
// //         throw new Error("addTensor input be 2d arrays")
// //     }

// //     const aValuesArray = new Float32Array(a.values)

// //     const aValuesGPUBuffer = device.createBuffer({
// //         mappedAtCreation: true,
// //         size: aValuesArray.byteLength,
// //         usage: GPUBufferUsage.STORAGE
// //     })
// //     new Float32Array(aValuesGPUBuffer.getMappedRange()).set(aValuesArray)
// //     aValuesGPUBuffer.unmap()

// //     const bValuesArray = new Float32Array(b.values)
// //     const bValuesGPUBuffer = device.createBuffer({
// //         mappedAtCreation: true,
// //         size: bValuesArray.byteLength,
// //         usage: GPUBufferUsage.STORAGE
// //     })
// //     new Float32Array(bValuesGPUBuffer.getMappedRange()).set(bValuesArray)
// //     bValuesGPUBuffer.unmap()

// //     const addResultGPUBuffer = device.createBuffer({
// //         size: aValuesArray.byteLength,
// //         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
// //     })

// //     const addReadGPUBuffer = device.createBuffer({
// //         size: aValuesArray.byteLength,
// //         usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
// //     })

// //     const bindGroupLayout = device.createBindGroupLayout({
// //         entries: [
// //             {
// //                 binding: 0,
// //                 visibility: GPUShaderStage.COMPUTE,
// //                 buffer: {
// //                     type: "read-only-storage"
// //                 }
// //             } as GPUBindGroupLayoutEntry,
// //             {
// //                 binding: 1,
// //                 visibility: GPUShaderStage.COMPUTE,
// //                 buffer: {
// //                     type: "read-only-storage"
// //                 }
// //             } as GPUBindGroupLayoutEntry,
// //             {
// //                 binding: 2,
// //                 visibility: GPUShaderStage.COMPUTE,
// //                 buffer: {
// //                     type: "storage"
// //                 }
// //             } as GPUBindGroupLayoutEntry
// //         ]
// //     })

// //     const bindGroup = device.createBindGroup({
// //         layout: bindGroupLayout,
// //         entries: [
// //             {
// //                 binding: 0,
// //                 resource: {
// //                     buffer: aValuesGPUBuffer
// //                 }
// //             },
// //             {
// //                 binding: 1,
// //                 resource: {
// //                     buffer: bValuesGPUBuffer
// //                 }
// //             },
// //             {
// //                 binding: 2,
// //                 resource: {
// //                     buffer: addResultGPUBuffer
// //                 }
// //             }
// //         ]
// //     })

// //     const computePipeline = device.createComputePipeline({
// //         layout: device.createPipelineLayout({
// //             bindGroupLayouts: [bindGroupLayout]
// //         }),
// //         compute: {
// //             module: device.createShaderModule({
// //                 code: // addWGSL 
// //                     `
// //                     // struct Shape {
// //                     //     shape: vec2<u32>
// //                     // };
                    
// //                     struct Values {
// //                         values: array<f32>
// //                     };
                    
// //                     // @group(0) @binding(0) var<storage, read> aShape:  Shape;
// //                     // @group(0) @binding(1) var<storage, read> bShape:  Shape;
// //                     @group(0) @binding(0) var<storage, read> aValues: Values;
// //                     @group(0) @binding(1) var<storage, read> bValues:  Values;
// //                     @group(0) @binding(2) var<storage, read_write> outValues:  Values;
                    
// //                     @compute @workgroup_size(1)
// //                     fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
// //                         let index: u32 = global_id.x;
// //                         outValues.values[index] = aValues.values[index] + bValues.values[index];
// //                     }
// //                 `
// //             }),
// //             entryPoint: "main"
// //         }
// //     })

// //     const commandEncoder = device.createCommandEncoder()
// //     const passEncoder = commandEncoder.beginComputePass()
// //     passEncoder.setPipeline(computePipeline)
// //     passEncoder.setBindGroup(0, bindGroup)
// //     passEncoder.dispatchWorkgroups(aValuesArray.length)
// //     passEncoder.end()

// //     commandEncoder.copyBufferToBuffer(addResultGPUBuffer, 0, addReadGPUBuffer, 0, aValuesArray.byteLength)
// //     device.queue.submit([commandEncoder.finish()])
// //     await addReadGPUBuffer.mapAsync(GPUMapMode.READ)

// //     return new Float32Array(addReadGPUBuffer.getMappedRange())
// // }

// // initDevice();