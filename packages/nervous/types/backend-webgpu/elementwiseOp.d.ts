import { BinaryOp, Tensor } from '../tensor';
export declare const elementwiseOP: (_a: Tensor, _b: Tensor | number, flag: BinaryOp) => Promise<Tensor>;
