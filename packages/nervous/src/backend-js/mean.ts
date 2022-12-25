import { Tensor } from "../tensor"
import { getValues } from "./getValues"
import { sum } from "./sum"

/** get the mean of all values */
export const mean = (a: Tensor): number => {
    return (getValues(sum(a)) / a.values.length)
}