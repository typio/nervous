import { ReduceOP, Tensor } from '../tensor'
import { reduceOP } from './reduceOP'

export const sum = async (a: Tensor, axis?: 0 | 1) => {
    return reduceOP(a, ReduceOP.sum, axis)
}
