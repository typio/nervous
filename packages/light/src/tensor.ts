type ValidArray =
    number[] |
    number[][] |
    number[][][] |
    number[][][][] |
    number[][][][][] |
    number[][][][][][]

// https://stackoverflow.com/a/68594661
const calcRank = (values: ValidArray | number) =>
    Array.isArray(values) ? 1 + Math.max(0, ...values.map(calcRank)) : 0;

const calcShape = (values: ValidArray) => {
    let shape: number[] = [];
    let subValues: ValidArray | number = values
    while (Array.isArray(subValues)) {
        shape.push(subValues.length);

        subValues = subValues[0];
    }
    return shape;
}

export class Tensor {
    readonly values: Float32Array
    readonly rank: number
    readonly shape: number[]

    constructor(values?: number | ValidArray) {
        if (values !== undefined) {
            if (!Array.isArray(values)) values = [values] // scalar number input as number[]
            let flatValues = values.flat(5)
            this.values = new Float32Array(flatValues)
            this.rank = calcRank(values)
            this.shape = calcShape(values)

        }
    }
}

export const scalar = (value: number) => {
    return new Tensor(value)
}

export const tensor1d = (values: number[]) => {
    return new Tensor(values)
}

export const tensor2d = (values: number[][]) => {
    return new Tensor(values)
}

export const tensor3d = (values: number[][][]) => {
    return new Tensor(values)
}

export const tensor4d = (values: number[][][][]) => {
    return new Tensor(values)
}

export const tensor5d = (values: number[][][][][]) => {
    return new Tensor(values)
}

export const tensor6d = (values: number[][][][][][]) => {
    return new Tensor(values)
}

export const zeroes = () => {

}

export const ones = () => {

}

export const random = () => {

}