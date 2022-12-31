import { Tensor } from "../tensor"
import { tensor } from "./tensor"

/** returns sum in Tensor of all tensor flatValues(), if 2d matrix axis can be specified: 0 for columns 1 for rows*/
export const sum = (a: Tensor, axis?: 0 | 1): Tensor => {
    if (a.rank() === 0) return a
    if (axis === 0) {
        if (a.rank() === 1) return a
        if (a.rank() > 2) throw new Error('Rank too high for column sum, rank() is >2')

        let newV = new Array(a.shape()[1]).fill(0)
        for (let i = 0; i < a.shape()[1]; i++) {
            for (let j = 0; j < a.shape()[0]; j++) {
                newV[i] += a.flatValues()[i + a.shape()[1] * j]

            }
        }
        return tensor(newV, [a.shape()[1]])

    } else if (axis === 1) {
        if (a.rank() > 2) throw new Error('Rank too high for row sum, rank() is >2')

        let newV = new Array(a.shape()[0]).fill(0)
        for (let i = 0; i < a.shape()[0]; i++) {
            for (let j = 0; j < a.shape()[1]; j++) {
                newV[i] += a.flatValues()[i * a.shape()[1] + j]

            }
        }
        return tensor(newV, [a.shape()[0], 1])

    } else {
        let sum = 0
        for (let i = 0; i < a.flatValues().length; i++)
            sum += a.flatValues()[i]
        return tensor(sum)
    }
}