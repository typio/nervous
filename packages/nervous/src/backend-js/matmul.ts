import { Tensor } from "../tensor"

/** create tensor of dot product */
export const matmul = (a: Tensor, m: Tensor) => {
    if (typeof m === 'number' || a.rank() === 0) {
        throw new Error("Please use Tensor.mul() for tensor scalar multiplication.")
    }

    // if 1d * 1d 
    if ((a.rank() === 1 && m.rank() === 1) || (a.rank() === 2 && m.rank() === 2 && a.shape()[0] === 1 && m.shape()[1] === 1)) {
        let newV: number = 0
        for (let i = 0; i < a.flatValues().length; i++)
            newV += a.values()[i] * m.values()[i]
        return new Tensor(newV)
    }

    // 1d * 2d  
    if ((a.rank() === 1 && m.rank() > 1) || (m.rank() === 1 && a.rank() > 1)) {
        if (a.rank() === 1) {
            let newV = (new Array(m.shape()[1])).fill(0)
            for (let i = 0; i < m.shape()[1]; i++) {
                for (let j = 0; j < a.shape()[0]; j++) {
                    newV[i] += a.values()[j] * m.values()[j * m.shape()[1] + i];
                }
            }

            return new Tensor(newV)
        } else {

        }
    }

    // 2d * 2d
    if (a.rank() === 2 && m.rank() === 2) {
        if (a.shape()[1] !== m.shape()[0])
            throw new Error("Tensors doesn't have compatible shapes for multiplication.")

        let A = a.values()
        let B = m.values()

        let newV = new Array(a.shape()[0])
        for (let i = 0; i < newV.length; i++) {
            newV[i] = new Array(m.shape()[1])
            for (let j = 0; j < newV[i].length; j++)
                newV[i][j] = 0
        }

        for (let i = 0; i < A.length; i++) {
            for (let j = 0; j < B[0].length; j++) {
                for (let k = 0; k < B.length; k++) {
                    newV[i][j] += A[i][k] * B[k][j]
                }
            }
        }

        return new Tensor(newV)
    }

    throw new Error("Tensor matmul on rank() > 2 tensors not yet supported.")
}