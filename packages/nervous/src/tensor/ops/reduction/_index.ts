import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'
import { runComputeShader } from '../../../webGPU/_index'

import { ReductionOp, Tensor } from '../../tensor'
import { gpuDevice } from '../../..'
import { flatLengthFromShape } from '../../tensorUtils'

export const reductionOp = (op: ReductionOp, a: Tensor, axis: -1 | 0 | 1 = -1): Tensor => {
    let resShape: number[]

    if (axis === -1) resShape = a.tensorShape
    if (axis === 0) resShape = [0, 0, 1, a.tensorShape[3]]
    if (axis === 1) resShape = [0, 0, a.tensorShape[2], 1]

    let resultSize: number
    if (axis === 0) {
        resultSize = (4 + a.tensorShape[3]) * Float32Array.BYTES_PER_ELEMENT
    }
    else if (axis === 1) {
        resultSize = (4 + a.tensorShape[2]) * Float32Array.BYTES_PER_ELEMENT
    }
    else {
        resultSize = (4 + 1) * Float32Array.BYTES_PER_ELEMENT
    }

    const num_iterations = axis === -1 ? flatLengthFromShape(a.tensorShape) : a.tensorShape[3 - axis];

    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(32, resultSize),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    runComputeShader(
        gpuDevice,
        [a.buffer, resultGPUBuffer],
        wgsl`
            struct Matrix {
                s: vec4<f32>,
                v: array<f32>
            };

            @group(0) @binding(0) var<storage, read> a: Matrix;
            @group(0) @binding(1) var<storage, read_write> o:  Matrix;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let axis = ${axis};

                if (global_id.x >= ${num_iterations}) {
                    return;
                }

                let col_length = u32(a.s[2]);
                let row_length = u32(a.s[3]);

                if (axis == 1) {
                    o.s = vec4(0,0,a.s[2],1);
                } else if (axis == 0) {
                    o.s = vec4(0,0,1,a.s[3]);
                } else {
                    o.s = vec4(0, 0, 0, 1);
                }

                #if ${op === ReductionOp.sum}
                    var sum: f32 = 0.0;

                    if (axis == 1) {
                        for (var i = 0u; i < row_length; i++) {
                            let idx = global_id.x * row_length + i;
                            sum += a.v[idx];
                        }
                        o.v[global_id.x] = sum;
                    } else if (axis == 0) {
                        for (var i = 0u; i < col_length; i++) {
                            let idx = u32(global_id.x + row_length * i);
                            sum += a.v[idx];
                        }
                        o.v[global_id.x] = sum;
                    } else {
                        // unfortunately this is single threaded
                        let length = u32(max(1., a.s.x) * max(1., a.s.y) * max(1., a.s.z) * max(1., a.s.w));
                        for (var i = 0u; i < length; i++){
                            sum += a.v[i];
                        }
                        o.v[0] = sum;
                    }


                #elif ${op === ReductionOp.argmax || op === ReductionOp.argmin}
                    var arg: u32 = 0u;

                    if (axis == 1) {
                        for (var i = 0u; i < row_length; i++) {
                            let idx = global_id.x * row_length + i;
                            if (a.v[idx]
                                ${op === ReductionOp.argmax ? '>' : '<'}
                                a.v[global_id.x * row_length + arg]){
                                arg = i;
                            }
                        }
                        o.v[global_id.x] = f32(arg);
                    } else if (axis == 0) {
                        for (var i = 0u; i < col_length; i++) {
                            let idx = u32(global_id.x + row_length * i);
                            if (a.v[idx]
                                ${op === ReductionOp.argmax ? '>' : '<'}
                                a.v[global_id.x + row_length * arg]){
                                arg = i;
                            }
                        }
                        o.v[global_id.x] = f32(arg);
                    } else {
                        let length = u32(max(1., a.s.x) * max(1., a.s.y) * max(1., a.s.z) * max(1., a.s.w));
                        for (var i = 0u; i < length; i ++){
                            if (a.v[i]
                                ${op === ReductionOp.argmax ? '>' : '<'}
                                a.v[arg]) {
                                arg = i;
                            }
                        }
                        o.v[0] = f32(arg);
                    }
                #endif
            }
        `,
        [Math.ceil(flatLengthFromShape(a.tensorShape) / 64), 1, 1]
    )

    return new Tensor(resultGPUBuffer, resShape)
}
