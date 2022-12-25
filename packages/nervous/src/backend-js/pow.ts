import { Tensor } from "../tensor"
import { broadcast } from "./_utils"

/** create tensor with relu done to all values  */
export const pow = (a: Tensor, exp: number) => {
    return broadcast(a, (x: number) => x ** exp)
}