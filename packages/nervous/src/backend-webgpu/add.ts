import { BinaryOp, Tensor } from "../tensor"
import { elementwiseOP } from '../backend-webgpu/elementwiseOp'

export const add = async (_a: Tensor, _b: Tensor | number) => {
    return elementwiseOP(_a, _b, BinaryOp.add)
}