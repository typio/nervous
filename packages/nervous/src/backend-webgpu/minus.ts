import { BinaryOp, Tensor } from "../tensor"
import { elementwiseOP } from './elementwiseOP'

export const minus = (_a: Tensor, _b: Tensor | number) => {
    return elementwiseOP(_a, _b, BinaryOp.minus)
}
