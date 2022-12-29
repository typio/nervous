import { Tensor } from "../tensor"
import { toArr } from "../tensorUtils"

/** return flat tensor values */
export const flatValues = (a: Tensor, decimals?: number) => {
    return toArr(a.data, decimals, 4)
}
