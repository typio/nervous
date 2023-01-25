import { BinaryOp, Tensor } from '../tensor'
import { elementwiseOP } from './elementwiseOP'

export const add = async (a: Tensor, b: Tensor | number) => {
    return elementwiseOP(a, b, BinaryOp.add)
}
