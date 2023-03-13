import { wgsl } from 'wgsl-preprocessor/wgsl-preprocessor.js'
import { flatLengthFromShape } from '../tensorUtils'

import { gpuDevice } from '..'
import { ReduceOP, Tensor } from '../tensor'

/** returns scalar sum in Tensor of all tensor values, in case of 2d matrix, axis can be specified for vector of sums: 0 for columns 1 for rows */
export const reduceOP = async (_a: Tensor, flag: ReduceOP, _axis?: 0 | 1): Promise<Tensor> => {
  let a = _a

  if (!a.usingGPUBuffer) a = await a.toGPU()

  // has value -1 if no axis is given
  let axis = _axis === undefined ? -1 : _axis

  let resShape: number[]

  if (axis === -1) resShape = a.webGPUBufferShape
  if (axis === 0) resShape = [0, 0, 1, a.webGPUBufferShape[3]]
  if (axis === 1) resShape = [0, 0, a.webGPUBufferShape[2], 1]

  const axisGPUBuffer = gpuDevice.createBuffer({
    mappedAtCreation: true,
    size: 32,
    usage: GPUBufferUsage.STORAGE,
  })
  new Int32Array(axisGPUBuffer.getMappedRange()).set(new Int32Array([axis]))
  axisGPUBuffer.unmap()

  let resSize: number

  if (axis === 0) resSize = (4 + a.webGPUBufferShape[3]) * Float32Array.BYTES_PER_ELEMENT
  else if (axis === 1) resSize = (4 + a.webGPUBufferShape[2]) * Float32Array.BYTES_PER_ELEMENT
  else resSize = (4 + 1) * Float32Array.BYTES_PER_ELEMENT

  const flagGPUBuffer = gpuDevice.createBuffer({
    mappedAtCreation: true,
    size: 32,
    usage: GPUBufferUsage.STORAGE,
  })
  new Uint32Array(flagGPUBuffer.getMappedRange()).set(new Uint32Array([flag]))
  flagGPUBuffer.unmap()

  const resultGPUBuffer = gpuDevice.createBuffer({
    size: Math.max(32, resSize),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const computePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: {
      module: gpuDevice.createShaderModule({
        code: wgsl`
                struct Matrix {
    s: vec4<f32>,
    v: array<f32>
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> axis:  i32;
@group(0) @binding(2) var<storage, read> flag:  u32;
@group(0) @binding(3) var<storage, read_write> o:  Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (axis == -1 && global_id.x >= 1) {
        return;
    } else if (global_id.x >= u32(a.s[3 - axis])) {
        return;
    }

    let col_length = u32(a.s[2]);
    let row_length = u32(a.s[3]);

    if (axis == 1) {
        o.s = vec4(0,0,a.s[2],1);
    } else if (axis == 0) {
        o.s = vec4(0,0,1,a.s[3]);
    } else {
        o.s = vec4(0, 0, 0, 1);
    }

    switch flag {
        case 0u: { // sum
            var sum :f32 = 0.0;

            if (axis == 1) {
                for (var i = 0u; i < row_length; i++) {
                    let idx = global_id.x * row_length + i;
                    sum += a.v[idx];
                }
                o.v[global_id.x] = sum;
            } else if (axis == 0) {
                for (var i = 0u; i < col_length; i++) {
                    let idx = u32(global_id.x + row_length * i);
                    sum += a.v[idx];
                }
                o.v[global_id.x] = sum;
            } else {
                // unfortunately this is single threaded
                let length = u32(max(1., a.s.x) * max(1., a.s.y) * max(1., a.s.z) * max(1., a.s.w));
                for (var i = 0u; i < length; i++){
                    sum += a.v[i];
                }
                o.v[0] = sum;
            }
        }

        case 1u: { // argmax
            var argmax: u32 = 0u;

            if (axis == 1) {
                for (var i = 0u; i < row_length; i++) {
                    let idx = global_id.x * row_length + i;
                    if (a.v[idx] > a.v[global_id.x * row_length + argmax]){
                        argmax = i;
                    }
                }
                o.v[global_id.x] = f32(argmax);
            } else if (axis == 0) {
                for (var i = 0u; i < col_length; i++) {
                    let idx = u32(global_id.x + row_length * i);
                    if (a.v[idx] > a.v[global_id.x + row_length * argmax]){
                        argmax = i;
                    }
                }
                o.v[global_id.x] = f32(argmax);
            } else {
                let length = u32(max(1., a.s.x) * max(1., a.s.y) * max(1., a.s.z) * max(1., a.s.w));
                for (var i = 0u; i < length; i ++){
                    if (a.v[i] > a.v[argmax]) {
                        argmax = i;
                    }
                }
                o.v[0] = f32(argmax);
            }
        }
        default: {

        }
    }
}
`
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
          buffer: axisGPUBuffer,
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
  passEncoder.dispatchWorkgroups(Math.ceil(flatLengthFromShape(a.webGPUBufferShape) / 64))
  passEncoder.end()

  gpuDevice.queue.submit([commandEncoder.finish()])
  return new Tensor(resultGPUBuffer, resShape)
}
