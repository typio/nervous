/// <reference types="@webgpu/types" />
export declare const createMappedBuffer: (device: GPUDevice, data: Float32Array | Uint32Array, usage: GPUBufferUsageFlags) => GPUBuffer;
export declare const runComputeShader: (gpuDevice: GPUDevice, buffers: GPUBuffer[], code: string, workgroup_size: number[]) => GPUBuffer;
