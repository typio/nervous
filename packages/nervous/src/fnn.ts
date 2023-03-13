import { gpuDevice } from '..'
import { Tensor } from "../tensor"
import { Activation } from '../layer'
import { random } from './random'
import { zeros } from './zeros'
import snips from './snips'

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
    const trainX = train[0].usingGPUBuffer ? train[0] : train[0].toGPU()
    const trainy = train[0].usingGPUBuffer ? train[1] : train[1].toGPU()
    const testX = test[0].usingGPUBuffer ? test[0] : test[0].toGPU()
    const testy = test[1].usingGPUBuffer ? test[1] : test[1].toGPU()

    const layersHyperParamArray = new Float32Array(params.layers.length)
    const otherHyperParamsArray = new Float32Array(10)
    const hyperParamBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: Math.max(32, hyperParamArray.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    new Float32Array(hyperParamBuffer.getMappedRange()).set(a.data)
    hyperParamBuffer.unmap()

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
    let fnnWGSL = wgsl`
    ${snips.matrixStruct}

    @group(0) @binding(0) var<storage, read> trainX:  Matrix;
    @group(0) @binding(1) var<storage, read> trainy:  Matrix;
    @group(0) @binding(2) var<storage, read> testX:  Matrix;
    @group(0) @binding(3) var<storage, read> testy:  Matrix;
    @group(0) @binding(4) var<storage, read_write> outMatrix:  Matrix;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

      let outIndex: vec2<u32> = vec2(global_id.x, global_id.y);
      var sum: f32 = 0.0;
      for (var i: u32 = 0u; i < u32(aMatrix.s.w); i = i + 1u) {
        let a: u32 = i + outIndex.x * u32(aMatrix.s.w);
        let b: u32 = outIndex.y + i * u32(mMatrix.s.w);
        sum += aMatrix.v[a] * mMatrix.v[b];
      }

      outMatrix.v[outIndex.y + outIndex.x * u32(mMatrix.s.w)] = sum;
    }
  `

    const resultParamBufferSize = Float32Array.BYTES_PER_ELEMENT * 10
    const resultBufferSize = Float32Array.BYTES_PER_ELEMENT * 10
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: resultBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    const computePipeline = gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
            module: gpuDevice.createShaderModule({
                code: fnnWGSL,
            }),
            entryPoint: 'main',
        },
    })

    const bindGroup = gpuDevice.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: trainX.webGPUBuffer },
            },
            {
                binding: 1,
                resource: { buffer: trainy.webGPUBuffer },
            },
            {
                binding: 2,
                resource: { buffer: testX.webGPUBuffer },
            },
            {
                binding: 3,
                resource: { buffer: testy.webGPUBuffer },
            },
            {
                binding: 4,
                resource: { buffer: resultGPUBuffer },
            },
        ],
    })

    const commandEncoder = gpuDevice.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(1)

    passEncoder.end()
    gpuDevice.queue.submit([commandEncoder.finish()])
}
