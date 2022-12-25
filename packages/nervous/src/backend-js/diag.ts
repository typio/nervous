import { Tensor } from "../tensor"

export const diag = (values: number[]) => {
    // TODO: think about adding custom dimensions or single number values input
    let vLen = values.length
    let m = new Array(vLen * vLen).fill(0)
    let mI = 0
    let vI = 0
    while (vI < vLen) {
        m[mI] = values[vI]
        mI += vLen + 1
        vI++
    }
    return new Tensor(m, [vLen, vLen])
}
