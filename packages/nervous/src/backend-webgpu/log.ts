import { ScalarElementwiseOP, Tensor } from '../tensor'
import { scalarElementwiseOP } from './scalarElementwiseOP'

/** create tensor of log on all values */
export const log = (_a: Tensor, base: number) => {
    return scalarElementwiseOP(_a, base, ScalarElementwiseOP.log)
}
