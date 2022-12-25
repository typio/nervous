import { Tensor } from "../tensor"
import { flatLengthFromShape } from "../tensorUtils"

export const fill = (shape: number | number[], value: number) => {
    if (shape.constructor === Array)
        return new Tensor(new Array(flatLengthFromShape(shape)).fill(value), shape)
    else // @ts-ignore
        return new Tensor(new Array(shape).fill(value))
}