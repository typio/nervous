import { Tensor, ScalarElementwiseOP } from '../tensor'
import { scalarElementwiseOP } from './scalarElementwiseOP'

/** returns tensor with elementwise max of old value vs input number */
export const applyMin = (a: Tensor, n: number) => {
    return scalarElementwiseOP(a, n, ScalarElementwiseOP.applyMin)
}
