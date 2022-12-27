import { Rank1To4Array, Tensor } from './tensor';
export declare let gpuDevice: null | GPUDevice;
/** Create one hot tensor of provided shape with 1 at provided index */
export declare const oneHot: (dim: number[] | number, index: number | number[]) => Tensor;
/** Create tensor with non-zero values on diagonals from a provided value array */
export declare const diag: (values: number[]) => Tensor;
/** Create identity matrix tensor, optional horizontal offset on values  */
export declare const eye: (dim: number[] | number, offset?: number) => Tensor;
declare const _default: {
    init: (userConfig?: {
        backend: string;
    }) => Promise<any>;
    Tensor: typeof Tensor;
    add: (a: Tensor, b: number | Tensor, axis?: number) => Tensor;
    applyMax: (a: Tensor, n: number) => Tensor;
    applyMin: (a: Tensor, n: number) => Tensor;
    argmax: (a: Tensor) => Tensor;
    argmin: (a: Tensor) => Tensor;
    diag: (values: number[]) => Tensor;
    div: (a: Tensor, b: number | Tensor, axis?: number) => Tensor;
    exp: (a: Tensor, base?: number) => Tensor;
    eye: (dim: number | number[], offset?: number) => Tensor;
    fNorm: (a: Tensor) => number;
    fill: (shape: number[], value: number) => Tensor;
    getFlatValues: (a: Tensor, decimals?: number) => any;
    getValues: (a: Tensor, decimals?: number) => any;
    getmax: (a: Tensor, axis?: 0 | 1) => Tensor;
    getmin: (a: Tensor, axis?: 0 | 1) => Tensor;
    gradientReLU: (a: Tensor) => any;
    log: (a: Tensor, base?: number) => Tensor;
    lpNorm: (a: Tensor, p?: number) => number;
    matmul: (a: Tensor, b: Tensor) => Tensor;
    mean: (a: Tensor) => number;
    minus: (a: Tensor, b: number | Tensor, axis?: number) => Tensor;
    mod: (a: Tensor, b: number | Tensor, axis?: number) => Tensor;
    mul: (a: Tensor, b: number | Tensor, axis?: number) => Tensor;
    oneHot: (dim: number | number[], index: number | number[]) => Tensor;
    ones: (shape: number | number[]) => Tensor;
    pow: (a: Tensor, exp: number) => Tensor;
    random: (shape: number[], min?: number, max?: number, integer?: boolean) => Tensor;
    randomNormal: (shape: number[], mean?: number, std?: number) => Tensor;
    reLU: (a: Tensor) => Tensor;
    reshape: (a: Tensor, shape: number[]) => any;
    scalar: (value: number) => Tensor;
    sigmoid: (a: Tensor) => Tensor;
    softmax: (a: Tensor) => Tensor;
    softplus: (a: Tensor) => Tensor;
    sum: (a: Tensor, axis?: 0 | 1) => Tensor;
    tensor: (values: number | Rank1To4Array, shape?: number[]) => Tensor;
    trace: (a: Tensor) => number;
    transpose: (a: Tensor) => any;
    zeros: (shape: number | number[]) => Tensor;
};
export default _default;
