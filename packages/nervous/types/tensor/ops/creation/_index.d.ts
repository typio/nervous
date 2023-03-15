import { Rank1To4Array, Tensor } from "../../tensor";
export declare const scalar: (value: number) => Tensor;
export declare const tensor: (value: number | Rank1To4Array) => Tensor;
export declare const fill: (shape: number[], value: number) => Tensor;
export declare const zeros: (shape: number[]) => Tensor;
export declare const ones: (shape: number[]) => Tensor;
