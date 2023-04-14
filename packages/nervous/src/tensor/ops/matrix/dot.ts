import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'
import { runComputeShader, createMappedBuffer } from '../../../webGPU/_index'

import { Tensor } from '../../tensor'
import { gpuDevice } from '../../..'
import { flatLengthFromShape, padShape } from '../../tensorUtils'

/** create tensor of dot product */
export const dot = (a: Tensor, m: Tensor) => {
    if (typeof m === 'number') throw new Error('Please use Tensor.mul() for tensor scalar multiplication.')

    let aShape = a.tensorShape.map((e) => (e === 0 ? 1 : e)) // replacing 0s helps recognize a row vector
    let mShape = m.tensorShape.map((e) => (e === 0 ? 1 : e))
    let resShape = [0, 0, aShape[2], mShape[3]]
    let resSize = (4 + aShape[2] * mShape[3]) * Float32Array.BYTES_PER_ELEMENT

    if (aShape[3] !== mShape[2]) throw new Error('Tensors are not compatible shapes for multiplication.')

    const resultBufferSize = Math.max(32, resSize)
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(resultBufferSize, 32),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    runComputeShader(
        gpuDevice,
        [a.buffer, m.buffer, resultGPUBuffer],
        wgsl`
            struct Matrix {
                s: vec4<f32>,
                v: array<f32>
            };

            @group(0) @binding(0) var<storage, read> aMatrix: Matrix;
            @group(0) @binding(1) var<storage, read> mMatrix:  Matrix;
            @group(0) @binding(2) var<storage, read_write> outMatrix:  Matrix;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= max(1, u32(aMatrix.s.z)) || global_id.y >= max(1, u32(mMatrix.s.w)) {
                    return;
                }

                outMatrix.s = vec4(0., 0., aMatrix.s.z, mMatrix.s.w);

                let outIndex: vec2<u32> = vec2(global_id.x, global_id.y);
                var sum: f32 = 0.0;
                for (var i: u32 = 0u; i < u32(aMatrix.s.w); i = i + 1u) {
                    let a: u32 = i + outIndex.x * u32(aMatrix.s.w);
                    let b: u32 = outIndex.y + i * u32(mMatrix.s.w);
                    sum += aMatrix.v[a] * mMatrix.v[b];
                }

                outMatrix.v[outIndex.y + outIndex.x * u32(mMatrix.s.w)] = sum;
            }
        `,
        [Math.ceil(resShape[2] / 8), Math.ceil(resShape[3] / 8), 1]
    )

    return new Tensor(resultGPUBuffer, resShape)
}
