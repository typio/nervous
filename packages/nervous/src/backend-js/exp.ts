import { Tensor } from "../tensor"
import { broadcast } from "./_utils"

/** create tensor of exponentials of all values on e, or given base  */
export const exp = (a: Tensor, base?: number) => {
    if (base === undefined)
        base = Math.E

    return broadcast(a, (x: number) => base ** x)
}