import { BinaryOp, Tensor } from '../tensor';
export declare const doOp: (first: number, second: number, op: BinaryOp) => number;
export declare const elementwiseOp: (m: Tensor, n: number | Tensor, op: BinaryOp) => Tensor;
export declare const broadcast: (a: Tensor, func: any) => Tensor;
