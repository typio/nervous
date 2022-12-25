import { Tensor } from "../tensor"
import { broadcast } from "./_utils"

/** create tensor of log on all values */
export const log = (a: Tensor) => {
    return broadcast(a, (x: number) => Math.log(x))
}