import { BinaryOp, Tensor } from "../tensor"
import { elementwiseOP } from "./elementwiseOP"

export const add = async (_a: Tensor, _b: Tensor | number) => {
    return elementwiseOP(_a, _b, BinaryOp.add)
}