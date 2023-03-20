import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'
import { runComputeShader, createMappedBuffer } from '../../../webGPU/_index'

import { BinaryOp, Tensor } from '../../tensor'
import { gpuDevice } from '../../..'
import { flatLengthFromShape, padShape } from '../../tensorUtils'

/** switch rows and columns of a >=2d Tensor */
export const transpose = async (a: Tensor) => {
    // its a scalar so we are done
    if (flatLengthFromShape(a.tensorShape) === 1) {
        return a
    }

    let resShape = [0, 0, a.tensorShape[3], a.tensorShape[2]]
    let resSize = (4 + flatLengthFromShape(a.tensorShape)) * Float32Array.BYTES_PER_ELEMENT

    const resultBufferSize = Math.max(32, resSize)
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(resultBufferSize, 32),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    runComputeShader(gpuDevice,
        [a.buffer, resultGPUBuffer],
        wgsl`
            struct Matrix {
                s: vec4<f32>,
                v: array<f32>
            };

            @group(0) @binding(0) var<storage, read> a: Matrix;
            @group(0) @binding(1) var<storage, read_write> o:  Matrix;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                if global_id.x >= max(1, u32(a.s.z)) || global_id.y >= max(1, u32(a.s.w)) {
                    return;
                }

                // TODO: try to speed up with shared memory
                o.s = vec4(0, 0, a.s[3], a.s[2]);
                o.v[global_id.x + u32(a.s[2]) * global_id.y] = a.v[global_id.x * u32(a.s[3]) + global_id.y];
            }
        `,
        [Math.ceil(resShape[2] / 8), Math.ceil(resShape[3] / 8), 1],
    )
    return new Tensor(resultGPUBuffer, resShape)
}


