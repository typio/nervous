import { Tensor } from "../tensor"
import { broadcast } from "./_utils"

/** create tensor with sigmoid done to all values  */
export const sigmoid = (a: Tensor) => {
    return broadcast(a, (n: number) => 1 / (1 + Math.E ** -n))
}