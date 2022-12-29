import { Tensor } from "../tensor"

export const scalar = (value: number) => {
    return new Tensor(value)
}
