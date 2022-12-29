import { Tensor } from "../tensor"
import { toNested, toArr } from "../tensorUtils"

// TODO: can this be accelerated using GPU?
export const values = (a: Tensor, decimals?: number) => {
    if (a.data.length === 5) return a.data[4]
    return toNested(toArr(a.data, decimals, 4), a.shape())
}