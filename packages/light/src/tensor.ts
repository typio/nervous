type Rank1To6Array = number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][]

const calcShape = (values: Rank1To6Array) => {
    let shape: number[] = []
    let subValues: Rank1To6Array | number = values
    while (Array.isArray(subValues)) {
        shape.push(subValues.length)

        subValues = subValues[0]
    }
    return shape
}

const lengthFromShape = (shape: number[]): number => {
    return shape.reduce((previousValue, currentValue) => previousValue * currentValue, 1)
}

const tensorToString = (values, shape) => {

}

export class Tensor {
    readonly values: Float32Array
    readonly rank: number
    readonly shape: number[]

    /**
     * Pass a nested array
     * ```ts
     * new Tensor([[1, 2],[3, 4]])
     * ```
     * Or pass a flat array and a shape
     * ```ts
     * new Tensor([1, 2, 3, 4], [2, 2])
     * ```
     */
    constructor(values: number | Rank1To6Array, shape?: number[]) {
        if (values !== undefined && shape === undefined) {
            if (!Array.isArray(values)) {
                // if scalar
                values = [values] // store scalar number as number[]
                this.shape = [1]
                this.rank = 0
            } else {
                this.shape = calcShape(values)
                this.rank = this.shape.length
            }
            let flatValues = values.flat(5)
            this.values = new Float32Array(flatValues)
        } else if (values !== undefined && shape !== undefined) {
            if (typeof values[0] !== 'number')
                throw new Error('If shape is given, values must be flat array, e.g. [1, 2, 3].')

            // @ts-ignore flatValues will always be number[]
            let flatValues: number[] = values

            if (lengthFromShape(shape) !== flatValues.length) throw new Error("Values don't fit into shape.")

            this.shape = shape
            this.rank = shape.length
            this.values = new Float32Array(flatValues)
        }
    }

    print() {
        tensorToString(this.values, this.shape)
    }

}

/**
 * Pass a value
 * ```ts
 * scalar(1)
 * ```
 */
export const scalar = (value: number) => {
    return new Tensor(value)
}

/**
 * Pass a nested array
 * ```ts
 * tensor1d([1, 2, 3])
 * ```
 */
export const tensor1d = (values: number[], shape?: number[]) => {
    return new Tensor(values, shape)
}

/**
 * Pass a nested array
 * ```ts
 * tensor2d([[1, 2], [3, 4]])
 * ```
 * Or pass a flat array and a shape
 * ```ts
 * tensor2d([1, 2, 3, 4], [2, 2])
 * ```
 */
export const tensor2d = (values: number[][] | number[], shape?: number[]) => {
    return new Tensor(values, shape)
}

/**
 * Pass a nested array
 * ```ts
 * tensor3d([[[1, 2, 3]]])
 * ```
 * Or pass a flat array and a shape
 * ```ts
 * tensor3d([1, 2, 3, 4], [2, 2])
 * ```
 */
export const tensor3d = (values: number[][][] | number[], shape?: number[]) => {
    return new Tensor(values, shape)
}

/**
 * Pass a nested array
 * ```ts
 * tensor4d([[1, 2],[3, 4]])
 * ```
 * Or pass a flat array and a shape
 * ```ts
 * tensor4d([1, 2, 3, 4], [2, 2])
 * ```
 */
export const tensor4d = (values: number[][][][] | number[], shape?: number[]) => {
    return new Tensor(values, shape)
}

/**
 * Pass a nested array
 * ```ts
 * tensor5d([[1, 2],[3, 4]])
 * ```
 * Or pass a flat array and a shape
 * ```ts
 * tensor5d([1, 2, 3, 4], [2, 2])
 * ```
 */
export const tensor5d = (values: number[][][][][] | number[], shape?: number[]) => {
    return new Tensor(values, shape)
}

/**
 * Pass a nested array
 * ```ts
 * tensor6d([[1, 2],[3, 4]])
 * ```
 * Or pass a flat array and a shape
 * ```ts
 * tensor6d([1, 2, 3, 4], [2, 2])
 * ```
 */
export const tensor6d = (values: number[][][][][][] | number[], shape?: number[]) => {
    return new Tensor(values, shape)
}

/**
 * Pass array of row number and column number
 * ```ts
 * eye([2, 2])
 * ```
 * Or a number for both
 * ```ts
 * eye(2); eye([2])
 * ```
 */
export const eye = (dim: number[] | number, offset?: number) => {
    let rowN: number, colN: number
    if (typeof dim === 'number') {
        dim = [dim]
    }
    rowN = dim[0]
    if (dim.length === 1)
        colN = dim[0]
    else
        colN = dim[1]

    let idx = offset ?? 0
    let values = new Array(rowN * colN).fill(0)
    while (idx < rowN * colN) {
        values[idx] = 1
        idx += rowN + 1
    }

    return new Tensor(values, [rowN, colN])
}

/**
 * Pass shape of matrix
 * ```ts
 * random([2, 2])
 * ```
 * And optionally min (inclusive), max (exclusive), and integer
 * ```ts
 * random([2, 2], 0, 10, true)
 * ```
 */
export const random = (shape: number[], min?: number, max?: number, integer?: boolean) => {
    if ((min !== undefined && max === undefined) || (max !== undefined && min === undefined))
        throw new Error('Must have either both min and max params or neither.')

    if (min !== undefined && max !== undefined)
        if (integer)
            return new Tensor(
                Array.from({ length: lengthFromShape(shape) }, () => Math.floor(Math.random() * (max - min) + min)),
                shape
            )
        else
            return new Tensor(
                Array.from({ length: lengthFromShape(shape) }, () => Math.random() * (max - min) + min),
                shape
            )

    return new Tensor(
        Array.from({ length: lengthFromShape(shape) }, () => Math.random()),
        shape
    )
}

export const fill = (shape: number[], value: number) => {
    return new Tensor(new Array(lengthFromShape(shape)).fill(value), shape)
}

export const zeroes = (shape: number[]) => {
    return fill(shape, 0)
}

export const ones = (shape: number[]) => {
    return fill(shape, 1)
}
