import { Rank1To4Array, Tensor } from "../tensor"

export const tensor = (values: number | Rank1To4Array, shape?: number[]) => {
    if (values.constructor === Array && values.length === 1) return new Tensor(values[0])
    return new Tensor(values, shape)
}

