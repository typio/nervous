import { gpuDevice } from '..'
import { Tensor } from '../tensor'
import { flatLengthFromShape } from '../tensorUtils'

import transposeWGSL from './transpose.wgsl?raw'

/** switch rows and columns of a >=2d Tensor */
export const transpose = async (_a: Tensor) => {
    let a = _a
    if (!_a.usingGPUBuffer) {
        a = await a.toGPU()
    }

    // its a scalar so we are done
    if (flatLengthFromShape(a.webGPUBufferShape) === 1) {
        return _a
    }

    let resShape = [0, 0, a.webGPUBufferShape[3], a.webGPUBufferShape[2]]
    let resSize = (4 + flatLengthFromShape(a.webGPUBufferShape)) * Float32Array.BYTES_PER_ELEMENT

    const resultBufferSize = Math.max(32, resSize)
    const resultGPUBuffer = gpuDevice.createBuffer({
        size: Math.max(resultBufferSize, 32),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    const computePipeline = gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
            module: gpuDevice.createShaderModule({
                code: transposeWGSL,
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
                resource: { buffer: resultGPUBuffer },
            },
        ],
    })

    const commandEncoder = gpuDevice.createCommandEncoder()
    const passEncoder = commandEncoder.beginComputePass()
    passEncoder.setPipeline(computePipeline)
    passEncoder.setBindGroup(0, bindGroup)
    passEncoder.dispatchWorkgroups(Math.ceil(resShape[2] / 8), Math.ceil(resShape[3] / 8))

    passEncoder.end()
    gpuDevice.queue.submit([commandEncoder.finish()])

    return new Tensor(resultGPUBuffer, resShape)
}
