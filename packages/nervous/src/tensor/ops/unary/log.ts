import { ScalarElementwiseOP, Tensor } from '../tensor'
import { scalarElementwiseOP } from './scalarElementwiseOP'

/** create tensor of log on all values */
export const log = (_a: Tensor, _base: number) => {
    let base = _base === undefined ? Math.E : _base
    return scalarElementwiseOP(_a, base, ScalarElementwiseOP.log)
}
