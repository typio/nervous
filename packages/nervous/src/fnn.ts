import { Tensor } from './tensor/tensor'
import { gpuDevice, random, zeros } from '.'
import { Activation } from './layer/layer'

import { createMappedBuffer, runComputeShader } from './webGPU/_index'

import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'

export type fnnHyperParams = {
    layers: number[],
    activation: Activation,
    LR: number,
    stepSize: number,
    batchSize: number,
    epochs: number,
    stopAtAccuracy?: number,
    stopAtLoss?: number,
    stopWhenAccuracyDrops?: number,
    stopWhenLossRises?: number,
    logEvery?: number,
}

export const fnn = async (train: Tensor[], test: Tensor[], params: fnnHyperParams) => {
    const trainX = train[0]
    const trainy = train[1]
    const testX = test[0]
    const testy = test[1]

    const resultBufferSize = Float32Array.BYTES_PER_ELEMENT * (4 + 10)
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    let w1 = random([4, 8])
    let b1 = zeros([8])
    let w2 = random([8, 3])
    let b2 = zeros([3])

    runComputeShader(
        gpuDevice,
        [
            trainX.buffer,
            trainy.buffer,
            testX.buffer,
            testy.buffer,

            w1.buffer,
            b1.buffer,
            // w2.buffer,
            // b2.buffer,

            resultGPUBuffer
        ],


        // pass
        // train and test X, y -- buffer
        // the result buffer to write to -- buffer
        //
        // how many iterations to do -- preproc
        // loss and accuracy to stop at -- preproc
        // layer count -- preproc
        // layer sizes -- preproc

        // return
        // iterations/batches? done
        // loss, and accuracy
        wgsl`
            struct Matrix {
                s: vec4<f32>,
                v: array<f32>
            }

            struct Output {
                s: vec4<f32>,
                loss: f32,
                accuracy: f32,
                step: f32
            }

            @group(0) @binding(0) var<storage, read> trainX:  Matrix;
            @group(0) @binding(1) var<storage, read> trainy:  Matrix;
            @group(0) @binding(2) var<storage, read> testX:  Matrix;
            @group(0) @binding(3) var<storage, read> testy:  Matrix;

            @group(0) @binding(4) var<storage, read_write> w1:  Matrix;
            @group(0) @binding(5) var<storage, read_write> b1:  Matrix;

            @group(0) @binding(6) var<storage, read_write> output:  Output;

            const input_size: i32 = 4;
            const hidden1_size: i32 = 10;
            const output_size: i32 = 3;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                output.s = vec4<f32>(0, 0, 1, 10);

                let a = trainX.v[0];
                let b = trainy.v[0];
                let c = testX.v[0];
                let d = testy.v[0];


                // do binaryOp on w1 and b1


                let e = w1.v[0];
                let f = b1.v[0];

                b1.v[0] = 9.8;

                output.loss = 0.2;
                output.accuracy = 4.0;
                output.step = 10.0;
            }
        `,
        [1, 1, 1]
    )

    await (new Tensor(b1.buffer, [1, 1, 1, 10])).print()
    await (new Tensor(resultGPUBuffer, [1, 1, 1, 10])).print()
}
