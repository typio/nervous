import { reduceOP } from './reduceOP'
import { Tensor, ReduceOP } from '../tensor'

export const argmax = async (a: Tensor, axis?: 0 | 1) => {
    return reduceOP(a, ReduceOP.argmax, axis)
}
