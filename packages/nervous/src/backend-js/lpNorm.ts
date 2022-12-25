import { Tensor } from "../tensor"

/** return the lp norm as number, default p is 2  */
export const lpNorm = (a: Tensor, p?: number): number => {
    if (p !== undefined) {
        let vals = a.values
        for (let i = 0; i < vals.length; i++) {
            vals[i] = vals[i] ** p
        }
        let sum = 0
        for (let i = 0; i < vals.length; i++) {
            sum += vals[i]
        }
        return sum ** (1 / p)
    } else if (p === Infinity) { // lp norm where p === Inf simplifies to value of largest magnitude el
        let max = -Infinity
        for (let i = 0; i < a.values.length; i++)
            max = a.values[i] > max ? a.values[i] : max
        return max
    } else {
        let vals = a.values
        for (let i = 0; i < vals.length; i++) {
            vals[i] = vals[i] ** 2
        }
        let sum = 0
        for (let i = 0; i < vals.length; i++) {
            sum += vals[i]
        }
        return sum ** (1 / 2)
    }
}