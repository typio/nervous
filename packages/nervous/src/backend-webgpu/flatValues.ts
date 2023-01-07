import { Tensor } from "../tensor"
import { toArr } from "../tensorUtils"

/** return flat tensor values */
export const flatValues = async (_a: Tensor, decimals?: number) => {
    let a: Tensor = _a.usingGPUBuffer ? await _a.toJS() : _a

    return toArr(a.data, decimals, 4)
}
