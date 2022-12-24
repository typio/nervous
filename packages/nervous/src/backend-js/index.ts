import { Rank1To4Array, Tensor } from "../tensor";
import { flatLengthFromShape } from "../tensorUtils";

const scalar = (value: number) => {
    return new Tensor(value)
}

const tensor = (values: number | Rank1To4Array, shape?: number[]) => {
    if (values.constructor === Array && values.length === 1) return new Tensor(values[0])
    return new Tensor(values, shape)
}

const fill = (shape: number | number[], value: number) => {
    if (shape.constructor === Array)
        return new Tensor(new Array(flatLengthFromShape(shape)).fill(value), shape)
    else // @ts-ignore
        return new Tensor(new Array(shape).fill(value))
}

const zeros = (shape: number | number[]) => {
    return fill(shape, 0)
}

const ones = (shape: number | number[]) => {
    return fill(shape, 1)
}

const random = (shape: number[], min?: number, max?: number, integer?: boolean) => {
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

const randomNormal = (shape: number[], mean?: number, std?: number) => {
    const randomNormalNumber = (mean: number, std: number) => {
        let u = 0;
        let v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    if (mean === undefined)
        mean = 0

    if (std === undefined)
        std = 1

    if (shape.constructor === Array)
        return new Tensor(
            Array.from({ length: flatLengthFromShape(shape) }, () => randomNormalNumber(mean, std)),
            shape
        )
    else
        return new Tensor(
            // @ts-ignore: I checked the type
            Array.from({ length: shape }, () => randomNormalNumber(mean, std))
        )
}

const add = (a: Tensor, b: Tensor) => {
    const result = new Float32Array(a.values.length);
    for (let i = 0; i < a.values.length; i++) {
        result[i] = a.values[i] + b.values[i];
    }
    return new Tensor(result, a.shape);
}



export default {
    scalar,
    tensor,
    fill,
    zeros,
    ones,
    random,
    randomNormal,

    add
}