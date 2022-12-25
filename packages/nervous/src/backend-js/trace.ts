import { Tensor } from "../tensor"

/** returns sum of diagonal elements as number */
export const trace = (a: Tensor) => {
    let shape0 = a.shape[0]
    if (a.rank === 2 && shape0 === a.shape[1]) {
        let sum = 0
        for (let i = 0; i < shape0; i++) {
            sum += a.values[i * shape0 + i]
        }
        return sum
    } else if (a.rank === 0) {
        return a.values[0]
    }
    throw new Error("Must be square 2d matrix")
}