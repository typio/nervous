import { Tensor } from "../tensor"

export const argmin = (a: Tensor) => {
    let aFlatValues = a.flatValues()
    let minI = 0
    if (a.rank() === 0)
        return 0
    else if (a.rank() === 1 || a.shape()[0] === 1) {
        for (let i = 0; i < aFlatValues.length; i++) {
            if (aFlatValues[i] < aFlatValues[minI])
                minI = i
        }
    } else
        throw new Error("Doesn't handle rank > 1")

    return minI
}