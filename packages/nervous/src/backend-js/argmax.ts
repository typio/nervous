import { Tensor } from '../tensor'

export const argmax = (a: Tensor, axis: number) => {
    let col_length = a.data[2]
    let row_length = a.data[3]

    let newV = []
    let flatValues = a.flatValues()

    if (axis == 1) {
        for (let i = 0; i < col_length; i++) {
            let argmax = 0
            for (let j = 0; j < row_length; j++) {
                if (flatValues[i * row_length + j] > flatValues[i * row_length + argmax]) {
                    argmax = j
                }
            }
            newV.push(argmax)
        }
        return new Tensor(newV, [col_length, 1])
    } else if (axis == 0) {
        for (let i = 0; i < row_length; i++) {
            let argmax = 0
            for (let j = 0; j < col_length; j++) {
                if (flatValues[i + row_length * j] > flatValues[i + row_length * argmax]) {
                    argmax = j
                    console.log(i, argmax, flatValues[i + row_length * argmax])
                }
            }
            newV.push(argmax)
        }
        return new Tensor(newV, [row_length])
    } else {
        let argmax = 0
        for (let i = 0; i < flatValues.length(); i++) {
            if (flatValues[i] > flatValues[argmax]) argmax = i
        }
        return new Tensor(argmax)
    }
}
