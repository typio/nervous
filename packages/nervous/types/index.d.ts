/// <reference types="@webgpu/types" />
import { Rank1To4Array, Tensor } from './tensor';
export declare let backend: any;
export declare let gpuAdapter: null | GPUAdapter;
export declare let gpuDevice: null | GPUDevice;
export declare const resetGPUDevice: () => Promise<void>;
declare const _default: {
    init: (userConfig?: {
        backend: string;
    }) => Promise<any>;
    webgpuAvailable: () => boolean;
    Tensor: typeof Tensor;
    fnn: (train: Tensor[], test: Tensor[], params: import("./backend-webgpu/fnn").fnnParams) => Promise<void>;
    scalar: (value: number) => Tensor;
    tensor: (values: number | Rank1To4Array, shape?: number[]) => Tensor;
    eye: (dim: number | number[], offset?: number) => Promise<Tensor>;
    diag: (values: number[]) => Promise<Tensor>;
    ones: (shape: number | number[]) => Promise<Tensor>;
    zeros: (shape: number | number[]) => Promise<Tensor>;
    fill: (shape: number[], value: number) => Promise<Tensor>;
    oneHot: (dim: number | number[], index: number | number[]) => Promise<Tensor>;
    random: (shape: number[], seed?: number, min?: number, max?: number, integer?: boolean) => Promise<Tensor>;
    randomNormal: (shape: number[], seed?: number, mean?: number, std?: number) => Promise<Tensor>;
};
export default _default;
