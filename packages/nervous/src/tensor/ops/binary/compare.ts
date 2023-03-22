import { gpuDevice } from '..'
import { Tensor } from '../tensor'
import { flatLengthFromShape } from '../tensorUtils'

import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

export const compare = async (_a: Tensor, _m: Tensor, _axis?: 0 | 1): Promise<Tensor> => {
    let a = _a
    let m = _m
    let axis: number

    if (!_a.usingGPUBuffer) a = a.toGPU()
    if (!_m.usingGPUBuffer) m = m.toGPU()

    if (_axis === undefined) axis = -1
    else axis = _axis

    let resShape: number[]

    if (
        a.webGPUBufferShape[0] !== m.webGPUBufferShape[0] || //
        a.webGPUBufferShape[1] !== m.webGPUBufferShape[1] || //
        a.webGPUBufferShape[2] !== m.webGPUBufferShape[2] || //
        a.webGPUBufferShape[3] !== m.webGPUBufferShape[3]
    )
        throw new Error('shapes must match')

    if (axis === -1) resShape = a.webGPUBufferShape
    if (axis === 0) resShape = [0, 0, 1, a.webGPUBufferShape[3]]
    if (axis === 1) resShape = [0, 0, a.webGPUBufferShape[2], 1]

    let resSize = Float32Array.BYTES_PER_ELEMENT * (4 + flatLengthFromShape(resShape))

    const axisGPUBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    new Int32Array(axisGPUBuffer.getMappedRange()).set(new Int32Array([axis]))
    axisGPUBuffer.unmap()

    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(resSize, 32),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    const computePipeline = gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
            module: gpuDevice.createShaderModule({
                code: wgsl`
struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> m:  Matrix;
@group(0) @binding(2) var<storage, read> axis: i32;
@group(0) @binding(3) var<storage, read_write> o: Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (axis == -1) {
        if global_id.x >= max(1, u32(a.s.z)) * max(1, u32(a.s.w)) {
            return;
        }

        o.s = a.s;
        var newV: f32 = 1.;
        if (a.v[global_id.x] != m.v[global_id.x]) {
            newV = 0;
        }
        o.v[global_id.x] = newV;
    } else if (axis == 0) { // columns are equal
        if global_id.x >= u32(a.s[3]) {
            return;
        }

        o.s = vec4(0., 0., 1., a.s[3]);
        var newV: f32 = 1.;
        for (var i: u32 = 0; i < u32(a.s[2]); i += 1) {
            let idx: u32 = global_id.x + u32(a.s[3]) * i;
            if (a.v[idx] != m.v[idx]) {
                newV = 0;
            }
        }
        o.v[global_id.x] = newV;
    } else if (axis == 1) { // rows are equal
        if global_id.x >= u32(a.s[2]) {
            return;
        }

        o.s = vec4(0., 0., a.s[2], 1.);
        var newV: f32 = 1.;
        for (var i: u32 = 0; i < u32(a.s[3]); i += 1) {
            let idx: u32 = global_id.x * u32(a.s[3]) + i;
            if (a.v[idx] != m.v[idx]) {
                newV = 0;
            }
        }
        o.v[global_id.x] = newV;
    }
}
                `,
            }),
            entryPoint: 'main',
        },
    })

    const bindGroup = gpuDevice.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: a.webGPUBuffer },
            },
            {
                binding: 1,
                resource: { buffer: m.webGPUBuffer },
            },
            {
                binding: 2,
                resource: { buffer: axisGPUBuffer },
            },
            {
                binding: 3,
                resource: { buffer: resultGPUBuffer },
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
