import { Tensor } from '../tensor';
/** returns tensor with elementwise max of old value vs input number */
export declare const applyMin: (a: Tensor, n: number) => Promise<Tensor>;
