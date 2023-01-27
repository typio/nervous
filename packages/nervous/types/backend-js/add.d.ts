import { Tensor } from '../tensor';
/** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
export declare const add: (a: Tensor, b: Tensor | number) => Tensor;
