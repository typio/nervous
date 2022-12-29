import { Tensor } from "../tensor"

/** get the mean of all values */
export const mean = (a: Tensor): number => {
    return a.sum().values() / a.flatValues().length
}