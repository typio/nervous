import { calcShape, flatLengthFromShape } from "./tensorUtils"

export type Rank1To4Array = Float32Array | number[] | number[][] | number[][][] | number[][][][]
export type BinaryOp = "add" | "sub" | "mul" | "div" | "mod"

export class Tensor {
    readonly values: Float32Array = new Float32Array(0)
    readonly rank: 0 | 1 | 2 | 3 | 4 = 0
    readonly shape: number[] = [0]

    constructor(values: number | Rank1To4Array, shape?: number[]) {
        if (values !== undefined && shape === undefined) {
            if (values.constructor === Number) { // if scalar
                values = [values] // store scalar number as number[]
                this.shape = [1]
                this.rank = 0
            } else {
                this.shape = calcShape(values)
                this.rank = this.shape.length as typeof this.rank
            }
            if (values.constructor === Array) {
                let flatValues = values.flat() as number[]
                this.values = new Float32Array(flatValues)
            } else if (values.constructor === Float32Array) {
                this.values = values
            }

        } else if (values !== undefined && shape !== undefined) {
            if (values.constructor === Array && values[0].constructor === Array)
                throw new Error('If shape is given, values must be flat array, e.g. [1, 2, 3].')

            // @ts-ignore: I checked the type
            let flatValues: number[] | Float32Array = values

            if (flatLengthFromShape(shape) !== flatValues.length)
                throw new Error("Values don't fit into shape.")

            this.shape = shape
            this.rank = shape.length as typeof this.rank // good ts? 🤔
            this.values = new Float32Array(flatValues)
        }
    }
}
