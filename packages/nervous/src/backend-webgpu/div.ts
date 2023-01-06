import { BinaryOp, Tensor } from "../tensor"
import { elementwiseOP } from './elementwiseOp'

export const div = async (_a: Tensor, _b: Tensor | number) => {
    return elementwiseOP(_a, _b, BinaryOp.div)
}