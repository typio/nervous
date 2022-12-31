import { Tensor } from "../tensor"

export const argmax = (a: Tensor) => {
    let aFlatValues = a.flatValues()
    let maxI = 0
    if (a.rank() === 0)
        return 0
    else if (a.rank() === 1) {
        for (let i = 0; i < aFlatValues.length; i++) {
            if (aFlatValues[i] > aFlatValues[maxI])
                maxI = i
        }
    } else
        throw new Error("Doesn't handle rank > 1")

    return maxI
}