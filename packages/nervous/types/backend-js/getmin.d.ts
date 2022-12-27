import { Tensor } from "../tensor";
/** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
export declare const getmin: (a: Tensor, axis?: 0 | 1) => number | Tensor;
