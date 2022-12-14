import { gpuDevice } from '..'
import { Tensor } from '../tensor'
import { flatLengthFromShape } from '../tensorUtils'

import randomWGSL from './random.wgsl?raw'

export const random = async (shape: number[], seed?: number, min?: number, max?: number, integer?: boolean) => {
	let shapeArray = new Float32Array(shape)

	let seedUInt: Uint32Array
	if (seed === undefined) seedUInt = new Uint32Array([Math.random() * 0xffffffff])
	else {
		if (seed > 1000000 || seed < -1000000) throw new Error('random() seed must be in range [-1,000,000, 1,000,000]')
		seedUInt = new Uint32Array([(1 / (0.001 + ((seed - -1000000) * 0.998) / 2000000)) * 0xffffffff]) // non-idomatic range map
	}

	let resultSize = flatLengthFromShape(shape)

	const shapeGPUBuffer = gpuDevice.createBuffer({
		mappedAtCreation: true,
		size: Math.max(32, shapeArray.byteLength),
		usage: GPUBufferUsage.STORAGE,
	})
	new Float32Array(shapeGPUBuffer.getMappedRange()).set(shape)
	shapeGPUBuffer.unmap()

	const seedGPUBuffer = gpuDevice.createBuffer({
		mappedAtCreation: true,
		size: seedUInt.byteLength,
		usage: GPUBufferUsage.STORAGE,
	})
	new Float32Array(seedGPUBuffer.getMappedRange()).set(seedUInt)
	seedGPUBuffer.unmap()

	const resultBufferSize = Math.max(32, Float32Array.BYTES_PER_ELEMENT * (4 + resultSize))
	const resultGPUBuffer = gpuDevice.createBuffer({
		size: resultBufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	})

	const readGPUBuffer = gpuDevice.createBuffer({
		size: resultBufferSize,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	})

	const computePipeline = gpuDevice.createComputePipeline({
		layout: 'auto',
		compute: {
			module: gpuDevice.createShaderModule({
				code: randomWGSL,
			}),
			entryPoint: 'main',
		},
	})

	const bindGroup = gpuDevice.createBindGroup({
		layout: computePipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: { buffer: shapeGPUBuffer },
			},
			{
				binding: 1,
				resource: { buffer: seedGPUBuffer },
			},
			{
				binding: 2,
				resource: { buffer: resultGPUBuffer },
			},
		],
	})

	const commandEncoder = gpuDevice.createCommandEncoder()
	const passEncoder = commandEncoder.beginComputePass()
	passEncoder.setPipeline(computePipeline)
	passEncoder.setBindGroup(0, bindGroup)
	passEncoder.dispatchWorkgroups(Math.ceil(resultSize / (64 * 4)))
	passEncoder.end()

	commandEncoder.copyBufferToBuffer(resultGPUBuffer, 0, readGPUBuffer, 0, resultBufferSize)
	gpuDevice.queue.submit([commandEncoder.finish()])
	await readGPUBuffer.mapAsync(GPUMapMode.READ)

	let result = new Float32Array(
		readGPUBuffer.getMappedRange().slice(0, Float32Array.BYTES_PER_ELEMENT * (4 + resultSize))
	)

	return new Tensor(result)
}
