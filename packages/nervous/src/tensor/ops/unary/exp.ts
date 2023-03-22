import { Tensor, ScalarElementwiseOP } from '../tensor'
import { scalarElementwiseOP } from './scalarElementwiseOP'

/** create tensor of exponentials of all values on e, or given base  */
export const exp = (a: Tensor, _base?: number) => {
    let base = _base === undefined ? Math.E : _base
    return scalarElementwiseOP(a, base, ScalarElementwiseOP.exp)
}
