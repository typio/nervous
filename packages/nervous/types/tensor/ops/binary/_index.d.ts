import { BinaryOp, Tensor } from '../../tensor';
export declare const binaryOp: (op: BinaryOp, a: Tensor, b: Tensor, axis?: 0 | 1) => Tensor;
export declare const elementwiseOP: (_a: Tensor, _b: Tensor | number, flag: BinaryOp) => Promise<Tensor>;
