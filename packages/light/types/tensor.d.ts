declare type Rank1To6Array = number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
export declare const flatLengthFromShape: (shape: number[]) => number;
export declare class Tensor {
    readonly values: Float32Array;
    readonly rank: 0 | 1 | 2 | 3 | 4 | 5 | 6;
    readonly shape: number[];
    constructor(values: number | Rank1To6Array, shape?: number[]);
    /** return nested tensor values */
    getValues(): any;
    /** return flat tensor values */
    getFlatValues(): any[];
    /** console.log nested tensor values */
    print(): void;
    /** Reshape tensor into provided shape */
    reshape(shape: number[]): Tensor;
    /** switch rows and columns of a >=2d Tensor */
    transpose(): Tensor;
    /** create tensor of dot product */
    dot(m: Tensor | number): Tensor;
    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul(m: Tensor | number): Tensor;
    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div(m: Tensor | number): Tensor;
    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add(a: number | Tensor): Tensor;
    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus(m: number | Tensor): Tensor;
    /** create tensor of exponentials of all values on e, or given base  */
    exp(base?: number): Tensor;
    /** returns sum of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum(axis?: 0 | 1): Tensor;
    /** returns tensor with elementwise max of old value vs input number */
    applyMax(n: number): Tensor;
    /** returns tensor with elementwise min of old value vs input number */
    applyMin(n: number): Tensor;
    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    getMax(axis?: 0 | 1): number | Tensor;
    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    getMin(axis?: 0 | 1): number | Tensor;
}
/**
 * Pass a value
 * ```ts
 * scalar(4)
 * ```
 */
export declare const scalar: (value: number) => Tensor;
/**
 * Pass a nested array
 * ```ts
 * tensor([[1,2],[3,4]])
 * ```
 * Or pass a flat array and a shape
 * ```ts
 * tensor([1, 2, 3, 4], [2, 2])
 * ```
 */
export declare const tensor: (values: number | Rank1To6Array, shape?: number[]) => Tensor;
/**
 * Pass array of row number and column number
 * ```ts
 * eye([2, 2])
 * ```
 * Or a number for both
 * ```ts
 * eye(2); eye([2])
 * ```
 */
export declare const eye: (dim: number[] | number, offset?: number) => Tensor;
/**
 * Pass shape of matrix
 * ```ts
 * random([2, 2])
 * ```
 * And optionally min (inclusive), max (exclusive), and integer
 * ```ts
 * random([2, 2], 0, 10, true)
 * ```
 */
export declare const random: (shape: number[], min?: number, max?: number, integer?: boolean) => Tensor;
/**
 * Pass shape of matrix
 * ```ts
 * fill([2, 2], 1)
 * ```
 */
export declare const fill: (shape: number | number[], value: number) => Tensor;
export declare const zeroes: (shape: number | number[]) => Tensor;
export declare const ones: (shape: number | number[]) => Tensor;
export {};
