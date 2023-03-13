/// <reference types="@webgpu/types" />
export type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][];
export declare enum BinaryOp {
    add = 0,
    minus = 1,
    mul = 2,
    div = 3,
    mod = 4
}
export declare enum ScalarElementwiseOP {
    log = 0,
    pow = 1,
    applyMax = 2,
    applyMin = 3,
    exp = 4
}
export declare enum ReduceOP {
    sum = 0,
    argmax = 1,
    argmin = 2
}
export declare class Tensor {
    /**  first 4 values are shape, most to least significant dimensions, left-padded with 0's, rest are tensor values */
    readonly data: Float32Array;
    readonly usingGPUBuffer: boolean;
    readonly webGPUBuffer: GPUBuffer;
    readonly webGPUBufferShape: number[];
    /** Construct tensor, pass value array, nested or un-nested, and optional shape if un-nested,
     * or pass raw Float32Array already in internal Tensor data form. */
    constructor(values: number | Rank1To4Array | Float32Array | GPUBuffer, shape?: number[]);
    toJS: () => Promise<Tensor>;
    toGPU: () => Tensor;
    select: (dim: number, index: number) => never;
    /** returns nested number array of tensor values, returns type number if scalar */
    values: (decimals?: number) => any;
    /** returns flat tensor values */
    flatValues: (decimals?: number) => any;
    print: (decimals?: number) => any;
    /** returns tensor rank */
    rank: () => any;
    /** returns tensor shape, scalar ➡️ shape [0], vector ➡️ [1, N] */
    shape: () => any;
    /** Reshape tensor into provided shape */
    reshape: (shape: number[]) => any;
    /** Repeat tensor along dimensions */
    repeat: (scales: number[]) => any;
    /** switch rows and columns of a >=2d Tensor */
    transpose: () => any;
    /** create tensor of dot product */
    dot: (m: Tensor) => any;
    inverse: () => never;
    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul: (m: Tensor | number) => any;
    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div: (d: Tensor | number) => any;
    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add: (b: Tensor | number) => any;
    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus: (s: Tensor | number) => any;
    /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
    mod: (m: Tensor | number) => any;
    /** create tensor with relu done to all values  */
    pow: (exp: number) => any;
    /** create tensor with sigmoid done to all values  */
    sigmoid: () => any;
    /** create tensor with softplus done to all values  */
    softplus: () => any;
    softmax: (dim: any) => any;
    /** create tensor with relu done to all values  */
    reLU: () => any;
    /** create tensor with relu done to all values  */
    gradientReLU: (b: Tensor) => any;
    /** create tensor of exponentials of all values on e, or given base  */
    exp: (base?: number) => any;
    /** create tensor of log on all values */
    log: (base: number) => any;
    /** get the mean of all values */
    mean: () => any;
    /** return the lp norm as number, default p is 2  */
    lpNorm: (p?: number) => any;
    /** return Frobenius Norm as number, represents the size of a matrix */
    fNorm: () => any;
    /** returns sum of diagonal elements as number */
    trace: () => any;
    /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum: (axis?: 0 | 1) => any;
    compare: (b: Tensor, axis: 0 | 1) => any;
    /** returns tensor with elementwise max of old value vs input number */
    applyMax: (n: number) => any;
    /** returns tensor with elementwise min of old value vs input number */
    applyMin: (n: number) => any;
    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    getmax: (axis?: 0 | 1) => any;
    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    getmin: (axis?: 0 | 1) => any;
    argmax: (axis?: 0 | 1) => any;
    argmin: (axis?: 0 | 1) => any;
}
