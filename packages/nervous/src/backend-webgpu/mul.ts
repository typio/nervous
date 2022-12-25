import { Tensor } from "../tensor"

/** create tensor of elementwise matrix multiplication, if using a "scalar" tensor put scalar in mul argument */
export const mul = (a: Tensor, m: Tensor | number, axis?: number) => {
    throw new Error('Method is not yet implemented in webgpu backend 😞')
}