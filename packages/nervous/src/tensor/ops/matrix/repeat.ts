import { gpuDevice } from '..'
import { Tensor } from '../tensor'
import { flatLengthFromShape, padShape } from '../tensorUtils'

import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

export const repeat = async (_a: Tensor, _scales: number[]): Promise<Tensor> => {
    let a = _a
    if (!a.usingGPUBuffer) a = await a.toGPU()

    let scales = padShape(_scales)
    let scalesFloatArray = new Float32Array(scales)
    const scalesGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: scalesFloatArray.byteLength,
        usage: GPUBufferUsage.STORAGE,
    })
    new Float32Array(scalesGPUBuffer.getMappedRange()).set(scalesFloatArray)
    scalesGPUBuffer.unmap()

    let resShape = a.webGPUBufferShape.map((e, i) => e * scales[i])
    console.log(resShape)

    let resSize = (4 + flatLengthFromShape(resShape)) * Float32Array.BYTES_PER_ELEMENT

    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(32, resSize),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    const computePipeline = gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
            module: gpuDevice.createShaderModule({
                code: wgsl`
                struct Matrix {
  shape: vec4<f32>,
  values: array<f32>
}

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> scales: vec4<f32>;
@group(0) @binding(2) var<storage, read_write> outMatrix: Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  outMatrix.shape = a.shape * scales;

  let li = global_id.x;

  i = li % dims[3];
  j = (li / dims[3]) % dims[2];
  k = (li / (dims[3] * dims[2])) % dims[1];
  l = (li / (dims[3] * dims[2] * dims[1])) % dims[0];

  outMatrix.values[i] = 1.;
}
`
            }),
            entryPoint: 'main',
        },
    })

    const bindGroup = gpuDevice.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: a.webGPUBuffer,
                },
            },
            {
                binding: 1,
                resource: {
                    buffer: scalesGPUBuffer,
                },
            },
            {
                binding: 2,
                resource: {
                    buffer: resultGPUBuffer,
                },
            },
        ],
    })

    const commandEncoder = gpuDevice.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(Math.ceil(flatLengthFromShape(resShape) / 64))
    passEncoder.end()

    gpuDevice.queue.submit([commandEncoder.finish()])
    return new Tensor(resultGPUBuffer, resShape)
}
