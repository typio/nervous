/// <reference types="@webgpu/types" />
export type Rank1To4Array = number[] | number[][] | number[][][] | number[][][][];
export declare enum BinaryOp {
    add = 0,
    minus = 1,
    mul = 2,
    div = 3,
    mod = 4,
    pow = 5,
    compare = 6
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
    readonly buffer: GPUBuffer;
    /** doesn't include shape, gradient shape is defined by first 4 of buffer */
    readonly gradientBuffer: GPUBuffer | undefined;
    /** util to decide if tensor OPs are legal inside JS while values are away on GPU */
    readonly tensorShape: number[];
    /** Construct tensor, pass value array, nested or un-nested, and optional shape if un-nested,
     * or pass raw Float32Array already in internal Tensor data form. */
    constructor(values: number | Rank1To4Array | Float32Array | GPUBuffer, shape?: number[]);
    /** returns [values, shape, gradientValues]*/
    toJS: () => Promise<[Float32Array, Float32Array, Float32Array]>;
    /** returns nested number array of tensor values, returns type number if scalar */
    values: (decimals?: number) => Promise<number | any[]>;
    /** returns flat tensor values */
    flatValues: (decimals?: number) => Promise<number | number[]>;
    print: (decimals?: number) => Promise<void>;
    /** returns tensor rank */
    rank: () => number;
    /** returns tensor shape, scalar ➡️ shape [0], vector ➡️ [1, N] */
    shape: () => number[];
    /** switch rows and columns of a >=2d Tensor */
    /** create tensor of dot product */
    /** Reshape tensor into provided shape */
    /** Repeat tensor along dimensions */
    /** create tensor with number a OR each value of a tensor
    * a added to each value of input tensor  */
    add: (b: Tensor | number) => Tensor;
    /** create tensor with number m OR each value of a tensor
    * m subtracted from each value of input tensor  */
    minus: (s: Tensor | number) => Tensor;
    /** create tensor of elementwise matrix multiplication,
    * if using a "scalar" tensor put scalar in mul argument */
    mul: (m: Tensor | number) => Tensor;
    /** create tensor of elementwise matrix division,
    * if using a "scalar" tensor put scalar in div argument */
    div: (d: Tensor | number) => Tensor;
    /** create tensor with number m OR each value of a tensor
    * m mod with each value of input tensor  */
    mod: (m: Tensor | number) => Tensor;
    pow: (exp: number) => Tensor;
    compare: (b: Tensor, axis: 0 | 1) => Tensor;
}
