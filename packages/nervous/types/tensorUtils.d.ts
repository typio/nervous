/** Convert FloatArray to Array (Array.from() is slow...) */
export declare const toArr: (floatArr: Float32Array, decimals?: number) => any[];
export declare const calcShape: (values: Rank1To4Array) => number[];
export declare const flatLengthFromShape: (shape: number[]) => number;
export declare const toNested: (values: number[], shape: number[]) => any;
