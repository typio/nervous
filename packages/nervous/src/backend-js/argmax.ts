import { Tensor } from "../tensor"

export const argmax = (a: Tensor) => {
    let maxI = 0
    if (a.rank === 0)
        return 0
    else if (a.rank === 1 || a.shape[0] === 1) {
        for (let i = 0; i < a.values.length; i++) {
            if (a.values[i] > a.values[maxI])
                maxI = i
        }
    } else
        throw new Error("Doesn't handle rank > 1")

    return maxI
}