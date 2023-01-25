import { ReduceOP, Tensor } from '../tensor';
/** returns scalar sum in Tensor of all tensor values, in case of 2d matrix, axis can be specified for vector of sums: 0 for columns 1 for rows */
export declare const reduceOP: (_a: Tensor, flag: ReduceOP, _axis?: 0 | 1) => Promise<Tensor>;
