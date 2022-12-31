import { BinaryOp, Tensor } from "../tensor"
import { toArr } from "../tensorUtils"

export const doOp = (first: number, second: number, op: BinaryOp) => {
    if (op === 'add')
        return first + second
    else if (op === 'sub')
        return first - second
    else if (op === 'mul')
        return first * second
    else if (op === 'div')
        return first / second
    else if (op === 'mod')
        return first % second
    else
        throw new Error("Invalid operation code passed")
}

export const elementwiseOp = (m: Tensor, n: number | Tensor, op: BinaryOp, axis?) => {
    let mShape = m.shape()
    let nShape = n.constructor === Number ? undefined : n.shape()

    let mFlatValues = m.flatValues()
    let nFlatValues = n.constructor === Number ? [n] : n.flatValues()

    let newV = mFlatValues

    if (n.constructor === Number || n.rank() === 0) {
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], nFlatValues[0], op)
        }
    } else if (axis === 1) {
        if (nShape[0] === 1 && nShape[1] !== mShape[1])
            throw new Error(`Second tensor of shape ${nShape} should equal first tensor shape ${mShape} on axis=1 but is ${mShape[1]}`)
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], nFlatValues[i % nFlatValues.length], op)
        }
        // } else if (axis === 0) {
        //     for (let i = 0; i < newV.length; i++) {
        //         newV[i] = doOp(newV[i], n.values()[i], op)
        //     }
    } else {
        if (mFlatValues.length !== nFlatValues.length)
            throw new Error("Tensors can't be of different sizes for elementwise operation")
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], nFlatValues[i], op)
        }
    }
    return new Tensor(newV, m.shape())
}

export const broadcast = (a: Tensor, func: (any)) => {
    let newV = []
    let aFlatValues = a.flatValues()
    for (let i = 0; i < aFlatValues.length; i++) {
        newV[i] = func(aFlatValues[i])
    }
    return new Tensor(newV, a.shape())
}