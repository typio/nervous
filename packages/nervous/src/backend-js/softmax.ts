import { Tensor } from "../tensor"
import { getFlatValues } from "./getFlatValues"
import { getmax } from "./getmax"
import { getValues } from "./getValues"
import { minus } from "./minus"
import { tensor } from "./tensor"

// return softmax
export const softmax = (a: Tensor) => {
    let minusMaxTensor: Tensor = a
    if (a.rank === 0 || a.rank === 1) {
        minusMaxTensor = new Tensor(
            getFlatValues(
                minus(
                    a,
                    getmax(a)
                )
            ),
            [1, a.values.length]
        )
    } else if (a.rank === 2) {
        let newV = new Array(a.shape[0])
        for (let i = 0; i < a.shape[0]; i++) {
            let row = new Tensor(
                getValues(a)[i]
            )
            newV[i] = getFlatValues(minus(row,
                getmax(row)
            ))
        }
        minusMaxTensor = tensor(newV)
    } else throw new Error(`Softmax only supports [0-2]d tensors, yours is ${a.rank}d`)

    let outputs = (new Array(minusMaxTensor.shape[0] * minusMaxTensor.shape[1]))

    for (let j = 0; j < minusMaxTensor.shape[0]; j++) {
        let eValues = [];
        for (let i = 0; i < minusMaxTensor.shape[1]; i++) {
            eValues.push(Math.E ** minusMaxTensor.values[j * minusMaxTensor.shape[1] + i]);
        }

        let eValuesSum = 0;
        for (let i = 0; i < eValues.length; i++) {
            eValuesSum += eValues[i];
        }

        for (let i = 0; i < eValues.length; i++) {
            outputs[j * minusMaxTensor.shape[1] + i] = eValues[i] / eValuesSum
        }
    }

    return new Tensor(outputs, minusMaxTensor.shape);
}