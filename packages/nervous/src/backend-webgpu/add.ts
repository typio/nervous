import addWGSL from './add.wgsl?raw'

import { Tensor } from "../tensor"
import { webgpuExecuteTTT } from "./_execTTT"

export const add = async (a: Tensor, b: Tensor) => {
    if (a.rank !== 2 || b.rank !== 2) {
        throw new Error("addTensor input be 2d arrays")
    }
    return webgpuExecuteTTT(a, b, addWGSL)
}