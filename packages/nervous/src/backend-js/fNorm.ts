import { Tensor } from "../tensor"

/** return Frobenius Norm as number, represents the size of a matrix */
export const fNorm = (a: Tensor) => {
    let fNorm = 0
    for (let i = 0; i < a.values.length; i++)
        fNorm += a.values[i] ** 2
    return Math.sqrt(fNorm)
}