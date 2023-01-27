import { Tensor } from '../tensor'
import { toArr } from '../tensorUtils'

export const rank = (a: Tensor) => {
    let shape: number[]

    shape = toArr(a.data.slice(0, 4))

    if (shape[3] === 1 && shape[2] === 0) return 0 // scalar

    let i = 3
    while (i > 0) {
        if (shape[i - 1] === 0) {
            break
        }
        i--
    }
    return 4 - i
}
