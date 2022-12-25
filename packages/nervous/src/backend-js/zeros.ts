import { fill } from "./fill"

export const zeros = (shape: number | number[]) => {
    return fill(shape, 0)
}