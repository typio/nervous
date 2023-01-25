import { Tensor } from "../tensor"
import { flatLengthFromShape } from "../tensorUtils"

export const random = (shape: number[], seed: number, min?: number, max?: number, integer?: boolean) => {
    if (seed !== undefined)
        console.warn('random() in js backend is not currently seedable, webgpu backend is.');


    if ((min !== undefined && max === undefined) || (max !== undefined && min === undefined))
        throw new Error('Must have either both min and max params or neither.')

    if (min !== undefined && max !== undefined)
        if (integer) {
            if (shape.constructor === Array)
                return new Tensor(
                    Array.from({ length: flatLengthFromShape(shape) }, () => Math.floor(Math.random() * (max - min) + min)),
                    shape
                )
            else
                return new Tensor(
                    // @ts-ignore: I checked the type
                    Array.from({ length: shape }, () => Math.floor(Math.random() * (max - min) + min))
                )
        } else {
            if (shape.constructor === Array)
                return new Tensor(
                    Array.from({ length: flatLengthFromShape(shape) }, () => Math.random() * (max - min) + min),
                    shape)
            else
                return new Tensor(
                    // @ts-ignore: I checked the type
                    Array.from({ length: shape }, () => Math.random() * (max - min) + min))
        }

    if (shape.constructor === Array)
        return new Tensor(
            Array.from({ length: flatLengthFromShape(shape) }, () => Math.random()),
            shape
        )
    else
        return new Tensor(
            // @ts-ignore: I checked the type
            Array.from({ length: shape }, () => Math.random())
        )
}