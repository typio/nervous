import { Tensor } from "../tensor"

export const rank = (a: Tensor) => {
    if (a.data.length === 2) return 0 // scalar
    let rank = 0
    for (let i = 0; i < 4; i++) {
        if (a.data[i] !== 0) rank++
        else return rank
    }
    return rank
}