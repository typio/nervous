export declare type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][];
export declare type BinaryOp = "Add" | "Sub" | "Mul" | "Div" | "Mod";
export declare class Tensor {
    readonly values: Float32Array;
    readonly rank: 0 | 1 | 2 | 3 | 4;
    readonly shape: number[];
    constructor(values: number | Rank1To4Array, shape?: number[]);
    select(dim: number, index: number): void;
    /** return nested number array of tensor values, returns type number if scalar */
    getValues(decimals?: number): any;
    /** return flat tensor values */
    getFlatValues(decimals?: number): any[];
    /** Reshape tensor into provided shape */
    reshape(shape: number[]): Tensor;
    /** switch rows and columns of a >=2d Tensor */
    transpose(): any;
    /** create tensor of dot product */
    matmul(m: Tensor | number): Tensor;
    inverse(): void;
    /** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
    mul(m: Tensor | number, axis?: number): Tensor;
    /** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
    div(d: Tensor | number, axis?: number): Tensor;
    /** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
    add(a: number | Tensor, axis?: number): Tensor;
    /** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
    minus(s: number | Tensor, axis?: number): Tensor;
    /** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
    mod(m: number | Tensor, axis?: number): Tensor;
    /** create tensor with relu done to all values  */
    pow(exp: number): Tensor;
    broadcast(func: (any)): Tensor;
    /** create tensor with sigmoid done to all values  */
    sigmoid(): Tensor;
    /** create tensor with softplus done to all values  */
    softplus(): Tensor;
    softmax(): Tensor;
    /** create tensor with relu done to all values  */
    reLU(): Tensor;
    /** create tensor with relu done to all values  */
    gradientReLU(): Tensor;
    /** create tensor of exponentials of all values on e, or given base  */
    exp(base?: number): Tensor;
    /** create tensor of log on all values */
    log(): Tensor;
    /** get the mean of all values */
    mean(): number;
    /** return the lp norm as number, default p is 2  */
    lpNorm(p?: number): number;
    /** return Frobenius Norm as number, represents the size of a matrix */
    fNorm(): number;
    /** returns sum of diagonal elements as number */
    trace(): number;
    /** returns sum in Tensor of all tensor values, if 2d matrix axis can be specified: 0 for columns 1 for rows*/
    sum(axis?: 0 | 1): Tensor;
    /** returns tensor with elementwise max of old value vs input number */
    applyMax(n: number): Tensor;
    /** returns tensor with elementwise min of old value vs input number */
    applyMin(n: number): Tensor;
    /** returns maximum vlaue in tensor, pass axis for tensor of maximums per an axis (only 2d, 0 for cols 1 for rows) */
    getmax(axis?: 0 | 1): any;
    /** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
    getmin(axis?: 0 | 1): any;
    argmax(): number;
    argmin(): number;
}
