import { fill } from '../backend-webgpu/fill'

export const ones = async (shape: number | number[]) => {
    return await fill(shape, 1)
}
