import { Tensor } from "../tensor"
import { toArr } from "../tensorUtils"

/** return flat tensor values */
export const getFlatValues = (a: Tensor, decimals?: number) => {
    return toArr(a.values, decimals)
}
