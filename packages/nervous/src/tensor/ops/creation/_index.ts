import { Rank1To4Array, Tensor } from "../../tensor"
import { createTensor, CreateMethod } from "./createTensor"

export const scalar = (value: number): Tensor => {
    return new Tensor(value)
}

export const tensor = (value: number | Rank1To4Array): Tensor => {
    return new Tensor(value)
}

export const fill = (shape: number[], value: number): Tensor => {
    return createTensor({
        method: CreateMethod.fill,
        shape,
        value
    })
}

export const zeros = (shape: number[]) => {
    return createTensor({
        method: CreateMethod.fill,
        shape,
        value: 0
    })
}

export const ones = (shape: number[]) => {
    return createTensor({
        method: CreateMethod.fill,
        shape,
        value: 1
    })
}
