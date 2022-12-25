import { Tensor } from "../tensor";
import { flatLengthFromShape } from "../tensorUtils";

export const randomNormal = (shape: number[], mean?: number, std?: number) => {
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