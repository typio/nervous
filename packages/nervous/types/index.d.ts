import { Rank1To4Array, Tensor } from './tensor';
export declare let backend: any;
export declare let gpuDevice: null | GPUDevice;
declare const _default: {
    init: (userConfig?: {
        backend: string;
    }) => Promise<any>;
    webgpuAvailable: () => boolean;
    Tensor: typeof Tensor;
    scalar: (value: number) => Tensor;
    tensor: (values: number | Rank1To4Array, shape?: number[]) => Tensor;
    eye: (dim: number | number[], offset?: number) => Tensor;
    diag: (values: number[]) => Tensor;
    ones: (shape: number | number[]) => Tensor;
    zeros: (shape: number | number[]) => Tensor;
    fill: (shape: number[], value: number) => Tensor;
    oneHot: (dim: number | number[], index: number | number[]) => Tensor;
    random: (shape: number[], seed?: number, min?: number, max?: number, integer?: boolean) => Tensor;
    randomNormal: (shape: number[], seed?: number, mean?: number, std?: number) => Tensor;
};
export default _default;
