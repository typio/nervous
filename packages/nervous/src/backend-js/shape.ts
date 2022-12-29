import { Tensor } from "../tensor"

export const shape = (a: Tensor): number[] => {
    if (a.data.length === 5) // is scalar
        return [a.data[4]]

    let shape = []
    for (let i = 0; i < 4; i++)
        if (a.data[i] !== 0)
            shape.push(a.data[i])

    if (shape.length === 1)
        shape = [1, ...shape]
    return shape
}