import { Tensor } from "../tensor"
import { toArr } from "../tensorUtils"

/** Reshape tensor into provided shape */
export const reshape = (a: Tensor, shape: number[]) => {
    return new Tensor(toArr(a.values), shape)
}