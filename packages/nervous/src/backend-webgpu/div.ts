import { Tensor } from "../tensor"

/** create tensor of elementwise matrix division, if using a "scalar" tensor put scalar in div argument */
export const div = (a: Tensor, d: Tensor | number, axis?: number) => {
    throw new Error('Method is not yet implemented in webgpu backend 😞')

}