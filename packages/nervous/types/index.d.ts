/// <reference types="@webgpu/types" />
export declare let gpuAdapter: null | GPUAdapter;
export declare let gpuDevice: null | GPUDevice;
export declare const webgpuAvailable: () => boolean;
export declare const init: () => Promise<void>;
export * from './tensor/_index';
export * from './fnn';
