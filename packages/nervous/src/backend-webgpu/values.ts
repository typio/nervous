import { Tensor } from "../tensor"
import { toNested, toArr } from "../tensorUtils"

export const values = async (_a: Tensor, decimals?: number) => {
    let a: Tensor = _a.usingGPUBuffer ? await _a.toJS() : _a

    if (a.data.length === 5) return a.data[4]
    return toNested(toArr(a.data, decimals, 4), await a.shape())
}