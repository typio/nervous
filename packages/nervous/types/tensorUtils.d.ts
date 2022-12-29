import { Rank1To4Array } from "./tensor";
/** Convert FloatArray to Array (Array.from() and slice are slow...) */
export declare const toArr: (floatArr: Float32Array, decimals?: number, startIndex?: number) => any[];
export declare const calcShape: (values: Rank1To4Array) => number[];
export declare const flatLengthFromShape: (shape: number[]) => number;
export declare const toNested: (values: number[], shape: number[]) => any;
