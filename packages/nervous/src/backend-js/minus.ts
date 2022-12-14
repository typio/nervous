import { Tensor } from "../tensor"
import { elementwiseOp } from "./_utils"

/** create tensor with number m OR each value of a tensor m subtracted from each value of input tensor  */
export const minus = (a: Tensor, s: number | Tensor, axis?: number) => {
    return elementwiseOp(a, s, 'sub', axis)
}