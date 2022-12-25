import { Tensor } from "../tensor"
import { toNested, toArr } from "../tensorUtils"

/** return nested number array of tensor values, returns type number if scalar */
export const getValues = (a: Tensor, decimals?: number) => {
    if (a.rank === 0) return a.values[0]
    return toNested(toArr(a.values, decimals), a.shape)
}