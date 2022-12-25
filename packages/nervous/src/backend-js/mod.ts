import { Tensor } from "../tensor"
import { elementwiseOp } from "./_utils"

/** create tensor with number m OR each value of a tensor m mod with each value of input tensor  */
export const mod = (a: Tensor, m: number | Tensor, axis?: number) => {
    return elementwiseOp(a, m, 'mod', axis)
}