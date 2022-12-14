import { Tensor } from "../tensor"
import { elementwiseOp } from "./_utils"

/** create tensor with number a OR each value of a tensor a added to each value of input tensor  */
export const add = (a: Tensor, b: Tensor | number, axis?: number) => {
    return elementwiseOp(a, b, 'add', axis)
}