import { Tensor } from '../tensor'
import { flatLengthFromShape, padShape } from '../tensorUtils'

import fillWGSL from './fill.wgsl?raw'

import { gpuDevice } from '..'

export const fill = async (_shape: number | number[], value: number) => {
	// @ts-ignore
	let shape: number[] = padShape(_shape)
	let shapeArray = new Float32Array(shape)

	let valueFloat32 = new Float32Array([value])

	let resultSize = flatLengthFromShape(shape)

	const shapeGPUBuffer = gpuDevice.createBuffer({
		mappedAtCreation: true,
		size: Math.max(32, shapeArray.byteLength),
		usage: GPUBufferUsage.STORAGE,
	})
	new Float32Array(shapeGPUBuffer.getMappedRange()).set(shapeArray)
	shapeGPUBuffer.unmap()

	const valueGPUBuffer = gpuDevice.createBuffer({
		mappedAtCreation: true,
		size: Math.max(32, valueFloat32.byteLength),
		usage: GPUBufferUsage.STORAGE,
	})
	new Float32Array(valueGPUBuffer.getMappedRange()).set(valueFloat32)
	valueGPUBuffer.unmap()

	const resultBufferSize = Math.max(32, Float32Array.BYTES_PER_ELEMENT * (4 + resultSize))
	const resultGPUBuffer = gpuDevice.createBuffer({
		size: resultBufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	})

	const readGPUBuffer = gpuDevice.createBuffer({
		size: resultBufferSize,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	})

	const bindGroupLayout = gpuDevice.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'read-only-storage',
				},
			},
			{
				binding: 1,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'read-only-storage',
				},
			},
			{
				binding: 2,
				visibility: GPUShaderStage.COMPUTE,
				buffer: {
					type: 'storage',
				},
			},
		],
	})

	const computePipeline = gpuDevice.createComputePipeline({
		// i got no idea why auto layout doesn't always work
		layout: gpuDevice.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		}),
		compute: {
			module: gpuDevice.createShaderModule({
				code: fillWGSL,
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
				resource: { buffer: valueGPUBuffer },
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
	passEncoder.dispatchWorkgroups(Math.ceil(resultSize / 64))
	passEncoder.end()

	commandEncoder.copyBufferToBuffer(resultGPUBuffer, 0, readGPUBuffer, 0, resultBufferSize)
	gpuDevice.queue.submit([commandEncoder.finish()])
	await readGPUBuffer.mapAsync(GPUMapMode.READ)

	let result = new Float32Array(
		readGPUBuffer.getMappedRange().slice(0, Float32Array.BYTES_PER_ELEMENT * (4 + resultSize))
	)

	return new Tensor(result)
}
