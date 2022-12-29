import { Tensor } from "../tensor"
import { tensor } from "./tensor"

/** switch rows and columns of a >=2d Tensor */
export const transpose = (a: Tensor) => {
    if (a.rank === 0)
        return a
    if (a.rank === 1) {
        let arr = []
        for (let i = 0; i < a.values.length; i++) {
            arr[i] = a.values[i]
        }
        return new Tensor(arr, [a.shape[0], 1])
    }
    if (a.rank === 2) {
        // idiomatic 👍
        const A = getValues(a)

        let newV = new Array(a.shape[1])
        for (let i = 0; i < newV.length; i++) {
            newV[i] = new Array(a.shape[0])
            for (let j = 0; j < newV[i].length; j++)
                newV[i][j] = 0
        }

        for (let i = 0; i < A.length; i++) {
            for (let j = 0; j < A[0].length; j++) {
                newV[j][i] = A[i][j]
            }
        }
        return tensor(newV)
    }
    throw new Error("Transpose on tensor of rank > 2 is not yet supported.")
}