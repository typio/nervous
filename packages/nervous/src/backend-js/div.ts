import { BinaryOp, Tensor } from '../tensor'
import { elementwiseOp } from './_utils'

/** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
export const div = (a: Tensor, d: Tensor | number, axis?: number) => {
    return elementwiseOp(a, d, BinaryOp.div)
}
