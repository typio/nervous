import { Tensor } from "../tensor"
import { broadcast } from "./_utils"

/** returns tensor with elementwise min of old value vs input number */
export const applyMin = (a: Tensor, n: number) => {
    return broadcast(a, (x: number) => (x < n) ? x : n)

}