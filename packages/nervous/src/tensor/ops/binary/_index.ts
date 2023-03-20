import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'
import { runComputeShader, createMappedBuffer } from '../../../webGPU/_index'

import { BinaryOp, Tensor } from '../../tensor'
import { gpuDevice } from '../../..'
import { flatLengthFromShape, padShape } from '../../tensorUtils'

export const binaryOp = (op: BinaryOp, a: Tensor, b: Tensor, axis?: 0 | 1): Tensor => {
    let aShape = padShape(a.tensorShape)
    let bShape = padShape(b.tensorShape)

    // 'Elementwise OPs only support (nd tensor and matching nd tensor, nd tensor and scalar, 2d matrix and row or column vector, and row or column vector on row or column vector).'

    let resShape = [
        Math.max(aShape[0], bShape[0]),
        Math.max(aShape[1], bShape[1]),
        Math.max(aShape[2], bShape[2]),
        Math.max(aShape[3], bShape[3]),
    ]

    let resSize = (4 + flatLengthFromShape(resShape)) * Float32Array.BYTES_PER_ELEMENT

    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(32, resSize),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    runComputeShader(
        gpuDevice,
        [a.buffer, b.buffer, resultGPUBuffer],
        wgsl`
            struct Matrix {
                s: vec4<f32>,
                v: array<f32>
            };

            @group(0) @binding(0) var<storage, read> a: Matrix;
            @group(0) @binding(1) var<storage, read> b:  Matrix;
            @group(0) @binding(2) var<storage, read_write> o:  Matrix;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                o.s = vec4(max(a.s.x, b.s.x), max(a.s.y, b.s.y), max(a.s.z, b.s.z), max(a.s.w, b.s.w));

                if (global_id.x >= u32(max(1., o.s.x) * max(1., o.s.y) * max(1., o.s.z) * max(1., o.s.w))) {
                    return;
                }

                let lI = f32(global_id.x);

                let aV = calculate_value(0, a.s, lI);
                let bV = calculate_value(1, b.s, lI);

                #if ${op === BinaryOp.add}
                    o.v[u32(lI)] = aV + bV;
                #elif ${op === BinaryOp.minus}
                    o.v[u32(lI)] = aV - bV;
                #elif ${op === BinaryOp.mul}
                    o.v[u32(lI)] = aV * bV;
                #elif ${op === BinaryOp.div}
                    o.v[u32(lI)] = aV / bV;
                #elif ${op === BinaryOp.mod}
                    o.v[u32(lI)] = aV % bV;
                #elif ${op === BinaryOp.pow}
                    o.v[u32(lI)] = pow(aV, bV);
                #elif ${op === BinaryOp.eq}
                    o.v[u32(lI)] = f32(aV == bV);
                #elif ${op === BinaryOp.gt}
                    o.v[u32(lI)] = f32(aV > bV);
                #elif ${op === BinaryOp.lt}
                    o.v[u32(lI)] = f32(aV < bV);
                #else
                    o.v[u32(lI)] = 0.;
                #endif
            }

            fn calculate_value(matrixId: u32, shape: vec4<f32>, lI: f32) -> f32 {
                var index: u32;

                if (shape.w <= 1. && shape.z <= 1.) {
                    index = 0u;
                } else if (shape.w <= 1.) {
                    index = u32(lI / o.s.w);
                } else if (shape.z <= 1.) {
                    index = u32(lI % o.s.w);
                } else {
                    index = u32(lI);
                }

                return select(b.v[index], a.v[index], matrixId == 0); // tensor.v[index];
            }
        `,
        [Math.ceil(flatLengthFromShape(resShape) / 64), 1, 1]
    )

    return new Tensor(resultGPUBuffer, resShape)
}

