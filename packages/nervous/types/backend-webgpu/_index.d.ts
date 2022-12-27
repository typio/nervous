declare const _default: {
    add: (a: import("../tensor").Tensor, b: import("../tensor").Tensor) => Promise<import("../tensor").Tensor>;
    applyMax: (a: import("../tensor").Tensor, n: number) => never;
    applyMin: (a: import("../tensor").Tensor, n: number) => never;
    argmax: (a: import("../tensor").Tensor) => never;
    argmin: (a: import("../tensor").Tensor) => never;
    diag: (values: number[]) => never;
    div: (a: import("../tensor").Tensor, d: number | import("../tensor").Tensor, axis?: number) => never;
    exp: (a: import("../tensor").Tensor, base?: number) => never;
    eye: (dim: number | number[], offset?: number) => import("../tensor").Tensor;
    fNorm: (a: import("../tensor").Tensor) => never;
    fill: (shape: number | number[], value: number) => import("../tensor").Tensor;
    getFlatValues: (a: import("../tensor").Tensor, decimals?: number) => any[];
    getValues: (a: import("../tensor").Tensor, decimals?: number) => never;
    getmax: (a: import("../tensor").Tensor, axis?: 0 | 1) => never;
    getmin: (a: import("../tensor").Tensor, axis?: 0 | 1) => never;
    gradientReLU: (a: import("../tensor").Tensor) => never;
    log: (a: import("../tensor").Tensor) => never;
    lpNorm: (a: import("../tensor").Tensor, p?: number) => number;
    matmul: (a: import("../tensor").Tensor, m: number | import("../tensor").Tensor) => never;
    mean: (a: import("../tensor").Tensor) => number;
    minus: (a: import("../tensor").Tensor, s: number | import("../tensor").Tensor, axis?: number) => never;
    mod: (a: import("../tensor").Tensor, m: number | import("../tensor").Tensor, axis?: number) => never;
    mul: (a: import("../tensor").Tensor, m: number | import("../tensor").Tensor, axis?: number) => never;
    oneHot: (dim: number | number[], index: number | number[]) => never;
    ones: (shape: number | number[]) => never;
    pow: (a: import("../tensor").Tensor, exp: number) => never;
    random: (shape: number[], min?: number, max?: number, integer?: boolean) => never;
    randomNormal: (shape: number[], mean?: number, std?: number) => never;
    reLU: (a: import("../tensor").Tensor) => never;
    reshape: (a: import("../tensor").Tensor, shape: number[]) => never;
    scalar: (value: number) => never;
    sigmoid: (a: import("../tensor").Tensor) => never;
    softmax: (a: import("../tensor").Tensor) => never;
    softplus: (a: import("../tensor").Tensor) => never;
    sum: (a: import("../tensor").Tensor, axis?: 0 | 1) => import("../tensor").Tensor;
    tensor: (values: number | import("../tensor").Rank1To4Array, shape?: number[]) => import("../tensor").Tensor;
    trace: (a: Tensor) => never;
    transpose: (a: import("../tensor").Tensor) => never;
    zeros: (shape: number | number[]) => never;
};
export default _default;
