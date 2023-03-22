import { flatLengthFromShape, padShape } from '../../tensorUtils'
import { runComputeShader, createMappedBuffer } from '../../../webGPU/_index'

import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

import { gpuDevice } from '../../..'

import { Rank1To4Array, Tensor } from "../../tensor"

type CommonArgs = {
    method: CreateMethod
    shape: number[]
}

type FillArgs = CommonArgs & {
    value: number
}

type DiagArgs = CommonArgs & {
    values: number[]
}

type RandomArgs = CommonArgs & {
    seed: number,
    min?: number,
    max?: number,
    integer: boolean
    mean?: number,
    std?: number,
}

type CreateTensorArgs = FillArgs | DiagArgs | RandomArgs

// core tensor creation methods, rest are derivatives of these
enum CreateMethod {
    fill,

    diag,

    random,

    // TODO:
    // oneHot,
}

export const scalar = (value: number): Tensor => {
    return new Tensor(value)
}

export const tensor = (value: number | Rank1To4Array, shape?: number[]): Tensor => {
    return new Tensor(value, shape)
}

export const fill = (shape: number[], value: number): Tensor => {
    return createTensor({
        method: CreateMethod.fill,
        shape,
        value
    })
}

export const zeros = (shape: number[]) => {
    return createTensor({
        method: CreateMethod.fill,
        shape,
        value: 0
    })
}

export const ones = (shape: number[]) => {
    return createTensor({
        method: CreateMethod.fill,
        shape,
        value: 1
    })
}

export const diag = (values: number[]) => {
    return createTensor({
        method: CreateMethod.diag,
        shape: [0, 0, values.length, values.length],
        values
    })
}

export const eye = (shape: number[]) => {
    return createTensor({
        method: CreateMethod.diag,
        shape,
        values: [1]
    })
}

export const random = (shape: number[], seed?: number, min?: number, max?: number, integer?: boolean) => {
    return createTensor({
        method: CreateMethod.random,
        shape,
        seed,
        min,
        max,
        integer: integer ?? false,
    })
}

export const randomNormal = (shape: number[], seed?: number, mean?: number, std?: number, integer?: boolean) => {
    return createTensor({
        method: CreateMethod.random,
        shape,
        seed,
        mean,
        std,
        integer: integer ?? false,
    })
}

export const createTensor = (args: CreateTensorArgs): Tensor => {
    let shape = padShape(args.shape)

    let values = new Float32Array(1);
    const maxParams = 6
    let wgslParams = new Float32Array(maxParams)

    let resultSize = flatLengthFromShape(shape)
    let workgroup_size = 0;

    switch (args.method) {
        case CreateMethod.fill:
            const fillArgs = args as FillArgs
            values = new Float32Array([fillArgs.value])
            workgroup_size = Math.ceil(resultSize / 64)
            break
        case CreateMethod.diag:
            const diagArgs = args as DiagArgs
            values = new Float32Array(diagArgs.values)
            workgroup_size = Math.ceil(resultSize / 64)
            break
        case CreateMethod.random:
            const randomArgs = args as RandomArgs
            wgslParams[0] = randomArgs.seed ?? Math.random() * 1e4
            wgslParams[1] = randomArgs.min ?? 0
            wgslParams[2] = randomArgs.max ?? 1
            wgslParams[3] = Number(randomArgs.integer) ?? 0
            wgslParams[4] = randomArgs.mean ?? 0
            wgslParams[5] = randomArgs.std ?? 1

            workgroup_size = Math.ceil(resultSize / (64 * 4))
            break
        default:
    }

    let resultBufferSize = Math.max(32, Float32Array.BYTES_PER_ELEMENT * (4 + resultSize))
    resultBufferSize = Math.ceil(resultBufferSize / 4) * 4;
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    runComputeShader(
        gpuDevice,
        [
            createMappedBuffer(gpuDevice, new Float32Array(shape), GPUBufferUsage.STORAGE),
            createMappedBuffer(gpuDevice, new Float32Array(values), GPUBufferUsage.STORAGE),
            createMappedBuffer(gpuDevice, new Float32Array(wgslParams), GPUBufferUsage.STORAGE),
            resultGPUBuffer,

        ],
        headerWGSL(values.length, maxParams) + (
            args.method === CreateMethod.fill
                ? fillWGSL
                : args.method === CreateMethod.diag
                    ? diagWGSL(values.length)
                    : args.method === CreateMethod.random
                        ? randomWGSL
                        : "")
        ,
        [workgroup_size, 1, 1]
    )
    return new Tensor(resultGPUBuffer, shape)
}

const headerWGSL = (valuesLength: number, maxParams: number): string => wgsl`
    struct Matrix {
        shape: vec4<f32>,
        values: array<f32>
    };

    @group(0) @binding(0) var<storage, read> shape: vec4<f32>;
    @group(0) @binding(1) var<storage, read> values: array<f32, ${valuesLength}>;
    @group(0) @binding(2) var<storage, read> params: array<f32, ${maxParams}>;
    @group(0) @binding(3) var<storage, read_write> outMatrix: Matrix;
`

const fillWGSL = wgsl`
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        // consume inputs to avoid dead code elimination
        let p = params[0];
        let v = values[0];

        outMatrix.shape = shape;

        outMatrix.values[global_id.x] = values[0];
    }
`

const diagWGSL = (valuesLength: number) => wgsl`
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let p = params[0];
        let v = values[0];

        outMatrix.shape = shape;

        if (global_id.x > u32(shape.w)) {
            return;
        }
        outMatrix.values[global_id.x * u32(shape.w + 1)] = values[global_id.x % ${valuesLength}];
    }
`

const randomWGSL = wgsl`
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let p = params[0];
        let v = values[0];

        outMatrix.shape = shape;

        // https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf
        // "quick and dirty LCG which has a period of 2^32"
        let seedInput = u32(params[0]);
        let seed = seedInput * (global_id.x + 1u) * 1099087573u;
        let outIndex: vec2<u32> = vec2(global_id.x, global_id.y);
        outMatrix.shape = shape;

        var z1: u32;
        var z2: u32;
        var z3: u32;
        var z4: u32;

        z1 = tau_step(seed, 13u, 19u, 12u, 429496729u);
        z2 = tau_step(seed, 2u, 25u, 4u, 429496729u);
        z3 = tau_step(seed, 3u, 19u, 17u, 429496729u);
        z4 = 1664525u * seed + 1013904223u;
        let r0: u32 = z1 ^ z2 ^ z3 ^ z4;

        z1 = tau_step(r0, 13u, 19u, 12u, 429496729u);
        z2 = tau_step(r0, 2u, 25u, 4u, 429496729u);
        z3 = tau_step(r0, 3u, 19u, 17u, 429496729u);
        z4 = 1664525u * r0 + 1013904223u;
        let r1: u32 = z1 ^ z2 ^ z3 ^ z4;

        z1 = tau_step(r1, 13u, 19u, 12u, 429496729u);
        z2 = tau_step(r1, 2u, 25u, 4u, 429496729u);
        z3 = tau_step(r1, 3u, 19u, 17u, 429496729u);
        z4 = 1664525u * r1 + 1013904223u;
        let r2: u32 = z1 ^ z2 ^ z3 ^ z4;

        z1 = tau_step(r2, 13u, 19u, 12u, 429496729u);
        z2 = tau_step(r2, 2u, 25u, 4u, 429496729u);
        z3 = tau_step(r2, 3u, 19u, 17u, 429496729u);
        z4 = 1664525u * r2 + 1013904223u;
        let r3: u32 = z1 ^ z2 ^ z3 ^ z4;

        outMatrix.values[outIndex.y + outIndex.x * 4u ] = f32(r0) * 2.3283064365387e-10;
        outMatrix.values[outIndex.y + outIndex.x * 4u + 1u ] = f32(r1) * 2.3283064365387e-10;
        outMatrix.values[outIndex.y + outIndex.x * 4u + 2u ] = f32(r2) * 2.3283064365387e-10;
        outMatrix.values[outIndex.y + outIndex.x * 4u + 3u ] = f32(r3) * 2.3283064365387e-10;
    }

    fn tau_step(z: u32, s1: u32, s2: u32, s3: u32, M: u32) -> u32 {
        let b = (((z << s1) ^ z) >> s2);
        let res = (((z & M) << s3) ^ b);
        return res;
    }
`
