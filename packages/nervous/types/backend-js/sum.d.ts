import { Tensor } from "../tensor";
/** returns sum in Tensor of all tensor flatValues(), if 2d matrix axis can be specified: 0 for columns 1 for rows*/
export declare const sum: (a: Tensor, axis?: 0 | 1) => Tensor;
