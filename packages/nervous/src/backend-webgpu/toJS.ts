import { gpuDevice } from '..'
import { Tensor } from '../tensor'
import { flatLengthFromShape, toArr } from '../tensorUtils'

export const toJS = async (a: Tensor): Promise<Tensor> => {
	let bufferSize = Math.max(32, a.webGPUBuffer.size)

	const readGPUBuffer = gpuDevice.createBuffer({
		size: bufferSize,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	})
	const commandEncoder = gpuDevice.createCommandEncoder()
	commandEncoder.copyBufferToBuffer(a.webGPUBuffer, 0, readGPUBuffer, 0, bufferSize)
	gpuDevice.queue.submit([commandEncoder.finish()])
	await readGPUBuffer.mapAsync(GPUMapMode.READ)

	let result = new Float32Array(readGPUBuffer.getMappedRange())

	// buffer may have been right padded to make minimum size, undo that
	let tensorSize = 4 + flatLengthFromShape(toArr(result.slice(0, 4)))
	if (bufferSize > tensorSize * Float32Array.BYTES_PER_ELEMENT) result = result.slice(0, tensorSize)

	return new Tensor(result)
}
