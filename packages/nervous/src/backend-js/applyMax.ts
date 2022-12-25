import { Tensor } from "../tensor"
import { broadcast } from "./_utils"

/** returns tensor with elementwise max of old value vs input number */
export const applyMax = (a: Tensor, n: number) => {
    return broadcast(a, (x: number) => (x > n) ? x : n)
}