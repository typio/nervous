import { Tensor } from '../tensor'
import { toArr } from '../tensorUtils'

export const shape = (a: Tensor): number[] => {
    // remove leading 0's in shape segement of data

    let shape: number[]
    if (a.usingGPUBuffer) {
        shape = a.webGPUBufferShape
    } else {
        shape = toArr(a.data.slice(0, 4))
    }

    let i = 0
    while (i < 3 && shape[i] === 0) i++

    return shape.slice(i, 4)
}
