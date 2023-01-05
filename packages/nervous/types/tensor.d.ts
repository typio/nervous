import "@webgpu/types";
export declare type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][];
export declare type BinaryOp = "add" | "sub" | "mul" | "div" | "mod";
export declare class Tensor {
    /**  first 4 values are shape (right padded 0s), rest are tensor values */
    readonly data: Float32Array;
    readonly usingGPUBuffer: boolean;
    readonly webGPUBuffer: GPUBuffer;
    /** Construct tensor, pass value array, nested or un-nested, and optional shape if un-nested,
     * or pass raw Float32Array already in internal Tensor data form. */
    constructor(values: number | Rank1To4Array | Float32Array | GPUBuffer, shape?: number[]);
    toJS: () => Tensor;
    toGPU: () => Tensor;
    select: (dim: number, index: number) => never;
    /** returns nested number array of tensor values, returns type number if scalar */
    values: (decimals?: number) => number[] | number;
    /** returns flat tensor values */
    flatValues: (decimals?: number) => number[];
    /** returns tensor rank */
    rank: () => any;
    /** returns tensor shape, scalar ➡️ shape [0], vector ➡️ [1, N] */
    shape: () => any;
    /** Reshape tensor into provided shape */
    reshape: (shape: number[]) => any;
    /** switch rows and columns of a >=2d Tensor */
    transpose: () => any;
    /** create tensor of dot product */
    matmul: (m: Tensor) => any;
    inverse: () => never;
    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul: (m: Tensor | number, axis?: number) => any;
    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div: (d: Tensor | number, axis?: number) => any;
    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add: (b: number | Tensor, axis?: number) => Tensor;
    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus: (s: number | Tensor, axis?: number) => any;
    /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
    mod: (m: number | Tensor, axis?: number) => any;
    /** create tensor with relu done to all values  */
    pow: (exp: number) => any;
    /** create tensor with sigmoid done to all values  */
    sigmoid: () => any;
    /** create tensor with softplus done to all values  */
    softplus: () => any;
    softmax: () => any;
    /** create tensor with relu done to all values  */
    reLU: () => any;
    /** create tensor with relu done to all values  */
    gradientReLU: () => any;
    /** create tensor of exponentials of all values on e, or given base  */
    exp: (base?: number) => any;
    /** create tensor of log on all values */
    log: () => any;
    /** get the mean of all values */
    mean: () => any;
    /** return the lp norm as number, default p is 2  */
    lpNorm: (p?: number) => number;
    /** return Frobenius Norm as number, represents the size of a matrix */
    fNorm: () => any;
    /** returns sum of diagonal elements as number */
    trace: () => any;
    /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum: (axis?: 0 | 1) => Tensor;
    /** returns tensor with elementwise max of old value vs input number */
    applymax: (n: number) => any;
    /** returns tensor with elementwise min of old value vs input number */
    applymin: (n: number) => any;
    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    getmax: (axis?: 0 | 1) => any;
    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    getmin: (axis?: 0 | 1) => any;
    argmax: () => any;
    argmin: () => any;
}
