import { Tensor } from "../tensor"
import { broadcast } from "./_utils"

/** create tensor with relu done to all values  */
export const reLU = (a: Tensor) => {
    return broadcast(a, (x: number) => x > 0 ? x : 0)
}