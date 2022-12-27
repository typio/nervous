export declare type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][];
export declare type BinaryOp = "add" | "sub" | "mul" | "div" | "mod";
export declare class Tensor {
    readonly values: Float32Array;
    readonly rank: 0 | 1 | 2 | 3 | 4;
    readonly shape: number[];
    constructor(values: number | Rank1To4Array, shape?: number[]);
}
