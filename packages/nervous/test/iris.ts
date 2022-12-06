import { test } from 'uvu'
import * as assert from 'uvu/assert'

import fs from 'fs'
import * as d3 from "d3";

import * as nv from '../src/index'


if (typeof fetch !== 'function') {
    global.fetch = require('node-fetch-polyfill');
}
// const csv = require('d3-fetch').csv;

const prepareData = async () => {
    const text = await d3.text("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
    let data = d3.csvParseRows(text)
    data = data.map(row => row.map((s, i) => i === 4 ? s : +s))
    data.unshift(['SL', 'SW', 'PL', 'PW', 'Species'])

    const inputs = data.map(row => row.slice(0, -1));
    const targets = data.map(row => row.slice(-1));

    return [inputs, targets]
}

const main = async () => {
    const [inputs, targets] = await prepareData()

    const inputTensor = nv.tensor(inputs[Math.floor(Math.random() * inputs.length - 1) + 1])
    const hiddenWeights = nv.randomNormal([4, 5])
    const hiddenBias = nv.randomNormal([5])
    const hiddenLayer = inputTensor.matMul(hiddenWeights).add(hiddenBias)

    const outputWeights = nv.randomNormal([5, 3])
    const outputBias = nv.randomNormal([3])
    const outputLayer = hiddenLayer.matMul(outputWeights).add(outputBias)

    console.log(outputLayer);

    // One-hot encode the target labels
    // const oneHotTargets = nv.oneHotEncode(targets);
}

main()


test('iris', () => {
})

test.run()