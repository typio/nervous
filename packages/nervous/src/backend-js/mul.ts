import { BinaryOp, Tensor } from '../tensor'
import { elementwiseOp } from './_utils'

/** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
export const mul = (a: Tensor, m: Tensor | number, axis?: number) => {
    return elementwiseOp(a, m, BinaryOp.mul)
}
