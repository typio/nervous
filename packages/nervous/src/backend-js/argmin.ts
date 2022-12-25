import { Tensor } from "../tensor"

export const argmin = (a: Tensor) => {
    let minI = 0
    if (a.rank === 0)
        return 0
    else if (a.rank === 1 || a.shape[0] === 1) {
        for (let i = 0; i < a.values.length; i++) {
            if (a.values[i] < a.values[minI])
                minI = i
        }
    } else
        throw new Error("Doesn't handle rank > 1")

    return minI
}