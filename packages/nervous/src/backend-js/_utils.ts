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
    let newV = toArr(m.values)
    if (typeof n === 'number') {
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], n, op)
        }
    } else if (n.rank === 0) {
        let scalarValue = n.values[0]
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], scalarValue, op)
        }
    } else if (axis === 1) {
        if ((n.rank === 1 && n.shape[0] !== m.shape[1]) || (n.shape[0] === 1 && n.shape[1] !== m.shape[1]))
            throw new Error(`Second tensor of shape ${n.shape} should equal first tensor shape on axis=1 but is ${m.shape[1]}`)
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], n.values[i % n.values.length], op)
        }
        // } else if (axis === 0) {
        //     for (let i = 0; i < newV.length; i++) {
        //         newV[i] = doOp(newV[i], n.values[i], op)
        //     }
    } else {
        if (m.values.length !== n.values.length)
            throw new Error("Tensors can't be of different sizes for elementwise operation")
        for (let i = 0; i < newV.length; i++) {
            newV[i] = doOp(newV[i], n.values[i], op)
        }
    }
    return new Tensor(newV, m.shape)
}

export const broadcast = (a: Tensor, func: (any)) => {
    let newV = []
    for (let i = 0; i < a.values.length; i++) {
        newV[i] = func(a.values[i])
    }
    return new Tensor(newV, a.shape)
}