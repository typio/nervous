import { Tensor } from "../tensor"
import { tensor } from "./tensor"

// return softmax
export const softmax = (a: Tensor) => {
    let minusMaxTensor: Tensor = a
    if (a.rank() === 0 || a.rank() === 1) {
        minusMaxTensor = tensor(a.minus(a.getmax()).flatValues(), [1, a.flatValues().length])
    } else if (a.rank() === 2) {
        let newV = new Array(a.shape()[0])
        for (let i = 0; i < a.shape()[0]; i++) {
            let row = tensor(a.values()[i])
            newV[i] = row.minus(row.getmax()).flatValues()
        }
        minusMaxTensor = tensor(newV)
    } else throw new Error(`Softmax only supports [0-2]d tensors, yours is ${a.rank()}d`)


    let outputs = (new Array(minusMaxTensor.shape()[0] * minusMaxTensor.shape()[1]))

    for (let j = 0; j < minusMaxTensor.shape()[0]; j++) {
        let eValues = [];
        for (let i = 0; i < minusMaxTensor.shape()[1]; i++) {
            eValues.push(Math.E ** minusMaxTensor.flatValues()[j * minusMaxTensor.shape()[1] + i]);
        }

        let eValuesSum = 0;
        for (let i = 0; i < eValues.length; i++) {
            eValuesSum += eValues[i];
        }

        for (let i = 0; i < eValues.length; i++) {
            outputs[j * minusMaxTensor.shape()[1] + i] = eValues[i] / eValuesSum
        }
    }

    return new Tensor(outputs, minusMaxTensor.shape());
}