import { ScalarElementwiseOP, Tensor } from '../tensor'
import { scalarElementwiseOP } from './scalarElementwiseOP'

/** create tensor with relu done to all values  */
export const reLU = (a: Tensor) => {
    return scalarElementwiseOP(a, 0, ScalarElementwiseOP.applyMax)
}
