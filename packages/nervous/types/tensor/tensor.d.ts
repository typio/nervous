/// <reference types="@webgpu/types" />
export type Rank1To4Array = number[] | number[][] | number[][][] | number[][][][];
export declare enum UnaryOp {
    log = 0,
    exp = 1,
    relu = 2,
    leakyRelu = 3,
    softmax = 4
}
export declare enum BinaryOp {
    add = 0,
    minus = 1,
    mul = 2,
    div = 3,
    mod = 4,
    pow = 5,
    eq = 6,
    gt = 7,
    lt = 8,
    gradientRelu = 9,
    gradientLeakyRelu = 10
}
export declare enum ScalarElementwiseOp {
    log = 0,
    pow = 1,
    applyMax = 2,
    applyMin = 3,
    exp = 4
}
export declare enum ReductionOp {
    sum = 0,
    argmax = 1,
    argmin = 2,
    max = 3,
    min = 4,
    mean = 5
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
    transpose: () => Promise<Tensor>;
    /** create tensor of dot product */
    dot: (m: Tensor) => Tensor;
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
    /** to compare rows, columns, etc. use compare() a reduction operation */
    eq: (b: Tensor) => Tensor;
    lt: (b: Tensor) => Tensor;
    gt: (b: Tensor) => Tensor;
    gradientRelu: (b: Tensor) => Tensor;
    gradientLeakyRelu: (b: Tensor) => Tensor;
    /** create tensor of exponentials of all values on e, or given base  */
    exp: (base?: number) => Tensor;
    /** create tensor of log on all values */
    log: (base: number) => Tensor;
    /** returns tensor with elementwise max of old value vs input number */
    /** returns tensor with elementwise min of old value vs input number */
    /** create tensor with relu done to all values  */
    relu: () => Tensor;
    /** create tensor with relu done to all values  */
    leakyRelu: () => Tensor;
    /** create tensor with sigmoid done to all values  */
    /** create tensor with softplus done to all values  */
    softmax: (dim: number) => Tensor;
    argmax: (axis?: 0 | 1) => Tensor;
    argmin: (axis?: 0 | 1) => Tensor;
    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    max: (axis?: 0 | 1) => Tensor;
    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    min: (axis?: 0 | 1) => Tensor;
    /** get the mean of all values */
    mean: (axis?: 0 | 1) => Tensor;
    /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum: (axis?: 0 | 1) => Tensor;
}
