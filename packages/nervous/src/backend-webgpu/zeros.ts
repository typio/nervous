import { fill } from '../backend-webgpu/fill'

export const zeros = async (shape: number | number[]) => {
    return await fill(shape, 0)
}
