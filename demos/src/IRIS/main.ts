import * as d3 from "d3";

import * as nv from "nervous"

const prepareData = async () => {
    const text = await d3.text("iris.data");
    let data = d3.csvParseRows(text).slice(0, -1) // has a weird 0th element
    data = data.sort(() => Math.random() - 0.5) // shuffle

    data = data.map(row => row.map((s, i) => i === 4 ? s : +s))

    const inputs = data.map(row => row.slice(0, -1));
    const targets = [].concat(...data.map(row => row.slice(-1)));

    let cutoffIdx = Math.floor(data.length * .8)
    let trainData = inputs.slice(0, cutoffIdx)
    let trainLabels = targets.slice(0, cutoffIdx)
    let testData = inputs.slice(cutoffIdx)
    let testLabels = targets.slice(cutoffIdx)
    return [trainData, trainLabels, testData, testLabels]
}

const forward = (input: nv.Tensor, W: nv.Tensor, b: nv.Tensor) => {
    let scores = input.matmul(W).add(b, 1)
    return scores
}

const oneHotEncode = (vocab, value) => {
    let oneHot = (new Array(vocab.length)).fill(0)
    oneHot[vocab.indexOf(value)] = 1

    return oneHot
}

const evaluate = (inputData: nv.Tensor, inputLabels: nv.Tensor, W: nv.Tensor, b: nv.Tensor) => {
    let scores = forward(inputData, W, b)
    let scoresArr = scores.getValues()
    let correctScoresArr = inputLabels.getValues()

    let correctN = 0
    for (let i = 0; i < scoresArr.length; i++) {
        if (nv.tensor(scoresArr[i]).argmax() === nv.tensor(correctScoresArr[i]).argmax())
            correctN++
    }

    return Number((correctN / scoresArr.length * 100).toFixed(2))
}

let step_count_el = document.getElementById("step")

let train_steps_input = document.getElementById("training_steps")
let training_steps = Number(train_steps_input?.value)
train_steps_input.addEventListener('change', () => { training_steps = train_steps_input.value });

const LR = Number(document.getElementById("learning_rate")?.value)


const main = async () => {
    let step_count = 0

    const STEP_SIZE = 1

    let W = nv.randomNormal([4, 3])
    let b = nv.zeros(3)

    let [trainData, trainLabels, testData, testLabels] = await prepareData()
    let vocab = Array.from(new Set(trainLabels.concat(testLabels)))

    trainLabels = trainLabels.map((t: any) => oneHotEncode(vocab, t))
    testLabels = testLabels.map((t: any) => oneHotEncode(vocab, t))


    trainData = nv.tensor(trainData)
    trainLabels = nv.tensor(trainLabels)
    testData = nv.tensor(testData)
    testLabels = nv.tensor(testLabels)
    let train_samples_n = trainData.shape[0]

    document.getElementById("init_train_acc").innerHTML = ""
    document.getElementById("init_test_acc").innerHTML = ""
    document.getElementById("final_train_acc").innerHTML = ""
    document.getElementById("final_test_acc").innerHTML = ""

    document.getElementById("init_train_acc").innerHTML = evaluate(trainData, trainLabels, W, b)
    document.getElementById("init_test_acc").innerHTML = evaluate(testData, testLabels, W, b)


    const fit = () => {
        let scores = forward(trainData, W, b)

        let probs = scores.softmax()

        let correct_logprobs = probs.mul(trainLabels).sum(1).log().mul(-1)

        let data_loss = correct_logprobs.sum().getValues() / train_samples_n
        let reg_loss = 0.5 * LR * W.mul(W).sum().getValues()
        let loss = data_loss + reg_loss
        // if (i % 200 === 0) {
        //     console.log(`Iteration ${i}: Loss ${loss}`)
        step_count++

        // }

        let dscores = probs.minus(trainLabels)
        dscores = dscores.div(train_samples_n)


        let dW = trainData.transpose().matmul(dscores)
        let db = dscores.sum(0)

        dW = dW.add(W.mul(LR))

        // perform a parameter update
        W = W.add(dW.mul(-1 * STEP_SIZE))
        b = b.add(db.mul(-1 * STEP_SIZE))

        if (step_count <= training_steps) {
            step_count_el.innerHTML = step_count
            requestAnimationFrame(fit)
        }
        else {
            document.getElementById("final_train_acc").innerHTML = evaluate(trainData, trainLabels, W, b)
            document.getElementById("final_test_acc").innerHTML = evaluate(testData, testLabels, W, b)

        }
    }
    requestAnimationFrame(fit)
}

const fit = () => {

}

document.getElementById('fit_btn')?.addEventListener("click", main)
