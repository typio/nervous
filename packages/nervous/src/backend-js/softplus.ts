import { Tensor } from "../tensor"
import { broadcast } from "./_utils"

/** create tensor with softplus done to all values  */
export const softplus = (a: Tensor) => {
    return broadcast(a, (n: number) => Math.log(1 + Math.E ** n))
}