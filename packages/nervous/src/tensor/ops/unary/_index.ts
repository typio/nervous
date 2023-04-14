import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'
import { runComputeShader } from '../../../webGPU/_index'

import { UnaryOp, Tensor } from '../../tensor'
import { gpuDevice } from '../../..'
import { flatLengthFromShape } from '../../tensorUtils'

interface Args {
    dim?: number,
    base?: number,
    value?: number,
}

export const unaryOp = (op: UnaryOp, a: Tensor,
    { dim = 1, base = 0, value = 0 }: Args = {}): Tensor => {
    let resSize = (4 + flatLengthFromShape(a.tensorShape)) * Float32Array.BYTES_PER_ELEMENT

    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(32, resSize),
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

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                const dim = ${dim};

                let row_n = u32(a.s[2]);
                let col_n = u32(a.s[3]);

                o.s = a.s;

                #if ${op === UnaryOp.softmax}
                    if (global_id.x >= u32(a.s[3 - dim])) {
                        return;
                    }
                    var sum :f32 = 0.0;
                    var max = -0x1p-126f;
                    if (dim == 1) {
                        for (var i = 0u; i < col_n; i++) {
                            let idx = global_id.x * col_n + i;
                            let v = a.v[idx];
                            if (v > max) {
                                max = v;
                            }
                        }

                        for (var i = 0u; i < col_n; i++) {
                            let idx = global_id.x * col_n + i;
                            sum += exp(a.v[idx] - max);
                        }

                        for (var i = 0u; i < col_n; i++) {
                            let idx = global_id.x * col_n + i;
                            o.v[idx] = exp(a.v[idx] - max) / sum;
                        }
                    } else if (dim == 0) {
                        for (var i = 0u; i < row_n; i++) {
                            let idx = global_id.x + col_n * i;
                            let v = a.v[idx];
                            if (v > max) {
                                max = v;
                            }
                        }

                        for (var i = 0u; i < row_n; i++) {
                            let idx = u32(global_id.x + col_n * i);
                            sum += exp(a.v[idx] - max);
                        }

                        for (var i = 0u; i < row_n; i++) {
                            let idx = u32(global_id.x + col_n * i);
                            o.v[idx] = exp(a.v[idx] - max) / sum;
                        }
                    }
                #elif ${op === UnaryOp.log}
                #elif ${op === UnaryOp.exp}
                    o.v[global_id.x] = exp(a.v[global_id.x]);
                #elif ${op === UnaryOp.relu}
                    o.v[global_id.x] = select(0.0, a.v[global_id.x], a.v[global_id.x] > 0.0);
                #elif ${op === UnaryOp.leakyRelu}
                    o.v[global_id.x] = select(0.01 * a.v[global_id.x], a.v[global_id.x], a.v[global_id.x] > 0.0);
                #elif ${op === UnaryOp.tril}
                    if (global_id.x >= u32(a.s[2] * a.s[3])) {
                        return;
                    }
                    let row = global_id.x / col_n;
                    let col = global_id.x % col_n; // - row * col_n;
                    o.v[global_id.x] = select(
                        a.v[global_id.x],
                        ${value},
                        col > row
                    );
                #endif
            }
        `,
        [Math.ceil(flatLengthFromShape(a.tensorShape) / 64), 1, 1]
    )
    return new Tensor(resultGPUBuffer, a.tensorShape)
}
