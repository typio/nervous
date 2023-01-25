import { gpuDevice } from '..'
import { Tensor } from '../tensor'

export const toGPU = async (a: Tensor) => {
    const aBuffer = gpuDevice.createBuffer({
        mappedAtCreation: true,
        size: Math.max(32, a.data.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    new Float32Array(aBuffer.getMappedRange()).set(a.data)
    aBuffer.unmap()

    return new Tensor(aBuffer, await a.shape())
}
