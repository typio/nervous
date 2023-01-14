import elementwiseOPWGSL from './elementwiseOP.wgsl?raw'

import { BinaryOp, Tensor } from '../tensor'
import { gpuDevice } from '..'
import { flatLengthFromShape, padShape } from '../tensorUtils'

export const elementwiseOP = async (_a: Tensor, _b: Tensor | number, flag: BinaryOp) => {
  let a = _a
  let b = _b.constructor === Tensor ? _b : new Tensor(_b as number)

  if (!a.usingGPUBuffer) a = await a.toGPU()
  if (!b.usingGPUBuffer) b = await b.toGPU()

  let aShape = padShape(a.webGPUBufferShape)
  let bShape = padShape(b.webGPUBufferShape)

  // 'Elementwise OPs only support (nd tensor and matching nd tensor, nd tensor and scalar, 2d matrix and row or column vector, and row or column vector on row or column vector).'

  const flagGPUBuffer = gpuDevice.createBuffer({
    mappedAtCreation: true,
    size: 32,
    usage: GPUBufferUsage.STORAGE,
  })
  new Uint32Array(flagGPUBuffer.getMappedRange()).set(new Uint32Array([flag]))
  flagGPUBuffer.unmap()

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

  const computePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: {
      module: gpuDevice.createShaderModule({
        code: elementwiseOPWGSL,
      }),
      entryPoint: 'main',
    },
  })

  const bindGroup = gpuDevice.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: a.webGPUBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: b.webGPUBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: flagGPUBuffer,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: resultGPUBuffer,
        },
      },
    ],
  })

  const commandEncoder = gpuDevice.createCommandEncoder()
  const passEncoder = commandEncoder.beginComputePass()
  passEncoder.setPipeline(computePipeline)
  passEncoder.setBindGroup(0, bindGroup)
  passEncoder.dispatchWorkgroups(Math.ceil(flatLengthFromShape(resShape) / 64))
  passEncoder.end()

  gpuDevice.queue.submit([commandEncoder.finish()])
  return new Tensor(resultGPUBuffer, resShape)
}
