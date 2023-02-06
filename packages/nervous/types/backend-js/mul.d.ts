import { Tensor } from '../tensor';
/** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
export declare const mul: (a: Tensor, m: Tensor | number, axis?: number) => Tensor;
