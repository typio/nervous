import { Tensor } from "../tensor"
import { tensor } from "./tensor"

/** returns minimum vlaue in tensor, pass axis for tensor of minimums per an axis (only 2d, 0 for cols 1 for rows)*/
export const getmin = (a: Tensor, axis?: 0 | 1) => {
    if (axis === undefined) {
        let currVal = a.values[0]
        let min = currVal
        for (let i = 0; i < a.values.length; i++) {
            currVal = a.values[i]
            if (currVal < min) min = currVal
        }
        return min
    }
    let newV = new Array(a.shape[1]).fill(Infinity)
    if (axis === 0) {
        if (a.rank > 2) throw new Error('Rank too high for column max, rank is >2')
        for (let i = 0; i < a.shape[1]; i++) {
            for (let j = 0; j < a.shape[0]; j++) {
                let currVal = a.values[i + a.shape[1] * j]
                let oldMin = newV[i]
                if (currVal < oldMin) newV[i] = currVal
            }
        }
        return tensor(newV, [a.shape[1]])
    } else if (axis === 1) {
        if (a.rank > 2) throw new Error('Rank too high for row max, rank is >2')
        let newV = new Array(a.shape[0]).fill(Infinity)
        for (let i = 0; i < a.shape[0]; i++) {
            for (let j = 0; j < a.shape[1]; j++) {
                let currVal = a.values[i * a.shape[1] + j]
                let oldMin = newV[i]
                if (currVal < oldMin) newV[i] = currVal
            }
        }
        return tensor(newV, [a.shape[0], 1])
    }
}