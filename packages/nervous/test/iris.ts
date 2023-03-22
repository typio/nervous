// import { test } from 'uvu'
// import * as assert from 'uvu/assert'

// import fs from 'fs'
// import * as d3 from "d3";

// import * as nv from '../src/index'

// if (typeof fetch !== 'function') {
//     global.fetch = require('node-fetch-polyfill');
// }

// const prepareData = async () => {
//     const text = await d3.text("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
//     // const text = await d3.text("/Users/tohuber/Desktop/dev/nervous/packages/nervous/test/iris.data");
//     let data = d3.csvParseRows(text).slice(0, -1) // has a weird 0th element
//     data = data.sort(() => Math.random() - 0.5) // shuffle

//     data = data.map(row => row.map((s, i) => i === 4 ? s : +s))

//     const inputs = data.map(row => row.slice(0, -1));
//     const targets = [].concat(...data.map(row => row.slice(-1)));

//     let cutoffIdx = Math.floor(data.length * .8)
//     let trainData = inputs.slice(0, cutoffIdx)
//     let trainLabels = targets.slice(0, cutoffIdx)
//     let testData = inputs.slice(cutoffIdx)
//     let testLabels = targets.slice(cutoffIdx)
//     return [trainData, trainLabels, testData, testLabels]
// }

// const forward = (input: nv.Tensor, W1: nv.Tensor, b1: nv.Tensor, W2: nv.Tensor, b2: nv.Tensor) => {
//     let hidden_layer = input.dot(W1).add(b1, 1).reLU()
//     let scores = hidden_layer.dot(W2).add(b2, 1)
//     return [hidden_layer, scores]
// }

// const oneHotEncode = (vocab, value) => {
//     let oneHot = (new Array(vocab.length)).fill(0)
//     oneHot[vocab.indexOf(value)] = 1

//     return oneHot
// }

// const evaluate = (inputData: nv.Tensor, inputLabels: nv.Tensor, W1: nv.Tensor, b1: nv.Tensor, W2: nv.Tensor, b2: nv.Tensor) => {
//     let [_, scores] = forward(inputData, W1, b1, W2, b2)
//     let scoresArr = scores.getValues()
//     let correctScoresArr = inputLabels.getValues()

//     let correctN = 0
//     for (let i = 0; i < scoresArr.length; i++) {
//         if (nv.tensor(scoresArr[i]).argmax() === nv.tensor(correctScoresArr[i]).argmax())
//             correctN++
//     }

//     return Number((correctN / scoresArr.length*100).toFixed(2))
// }

// const main = async () => {
//     const STEP_SIZE = 1
//     const LR = 1E-2

//     const H = 6
//     let W1 = nv.randomNormal([4, H])
//     let b1 = nv.zeros(H)
//     let W2 = nv.randomNormal([H, 3])
//     let b2 = nv.zeros(3)

//     let [trainData, trainLabels, testData, testLabels] = await prepareData()
//     let vocab = Array.from(new Set(trainLabels.concat(testLabels)))
//     console.log(vocab);

//     trainLabels = trainLabels.map((t: any) => oneHotEncode(vocab, t))
//     testLabels = testLabels.map((t: any) => oneHotEncode(vocab, t))

//     // console.log('Initial test acc:', evaluate(testData, testLabels));

//     trainData = nv.tensor(trainData)

//     trainLabels = nv.tensor(trainLabels)
//     testData = nv.tensor(testData)
//     testLabels = nv.tensor(testLabels)
//     let train_samples_n = trainData.shape[0]

//     console.log("Initial train acc:", evaluate(trainData, trainLabels, W1, b1, W2, b2));
//     console.log("Initial test acc:", evaluate(testData, testLabels, W1, b1, W2, b2));

//     for (let i = 0; i <= 1000; i++) {
//         let [hidden_layer, scores] = forward(trainData, W1, b1, W2, b2)

//         let probs = scores.softmax()

//         let correct_logprobs = probs.mul(trainLabels).sum(1).log().mul(-1)

//         let data_loss = correct_logprobs.sum().getValues() / train_samples_n
//         let reg_loss = 0.5 * LR * W1.mul(W1).sum().getValues() + 0.5 * LR * W2.mul(W2).sum().getValues()
//         let loss = data_loss + reg_loss
//         if (i % 200 === 0) {
//             // console.log(data_loss);
//             // console.log(reg_loss);
//             // console.clear()
//             console.log(`Iteration ${i}: Loss ${loss}`)
//         }

//         let dscores = probs.minus(trainLabels) // subtracts 1 from every correct label index
//         dscores = dscores.div(train_samples_n)

//         let dW2 = hidden_layer.transpose().dot(dscores)
//         let db2 = dscores.sum(0) // axis = 0
//         let dhidden = dscores.dot(W2.transpose())
//         dhidden = dhidden.reLU()
//         // finally into W,b
//         let dW1 = trainData.transpose().dot(dhidden)
//         let db1 = dhidden.sum(0)

//         // add regularization gradient contribution
//         dW2 = dW2.add(W2.mul(LR))
//         dW1 = dW1.add(W1.mul(LR))

//         // perform a parameter update
//         W1 = W1.add(dW1.mul(-1 * STEP_SIZE))
//         b1 = b1.add(db1.mul(-1 * STEP_SIZE))
//         W2 = W2.add(dW2.mul(-1 * STEP_SIZE))
//         b2 = b2.add(db2.mul(-1 * STEP_SIZE))
//         // process.stdout.write(`${i},`);
//     }
//     // console.clear()
//     console.log("Final train acc:", evaluate(trainData, trainLabels, W1, b1, W2, b2));
//     console.log("Final test acc:", evaluate(testData, testLabels, W1, b1, W2, b2));
// }

// main()
