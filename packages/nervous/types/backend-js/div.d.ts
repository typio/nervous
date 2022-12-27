import { Tensor } from "../tensor";
/** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
export declare const div: (a: Tensor, d: Tensor | number, axis?: number) => Tensor;
