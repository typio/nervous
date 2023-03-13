<script lang="ts">
  import nv from "nervous";
  import Katex from "$lib/components/Katex.svelte";

  import iris_raw_data from "./iris.data?raw";
  import { browser } from "$app/environment";
  import type { Tensor } from "nervous/types/tensor";
  import type { fnnParams } from "nervous/types/backend-webgpu/fnn";


  let stop_signal = true;
  let main = () => {};
  let step_count = 0;
  let init_train_acc = "";
  let final_train_acc = "";
  let init_test_acc = "";
  let final_test_acc = "";
  let fit_btn_text = "▶";
  let backend = "webgpu";

  let training_steps = 100;
  let LR = 0.001;


  const prepareData = async () => {
    let data = iris_raw_data.split("\n");
    data = data.filter((row) => (row === "" ? false : true)); // remove empty lines
    data = data.sort(() => Math.random() - 0.5); // shuffle

    let mixed_type_data = data.map((row) =>
      row.split(",").map((s, i) => (i === 4 ? s : +s))
    );

    const inputs = mixed_type_data.map((row) => row.slice(0, -1));

    const targets = mixed_type_data.map((row) => row.slice(-1)[0]);

    let cutoffIdx = Math.floor(data.length * 0.8);
    let trainData = inputs.slice(0, cutoffIdx);
    let trainLabels = targets.slice(0, cutoffIdx);
    let testData = inputs.slice(cutoffIdx);
    let testLabels = targets.slice(cutoffIdx);

    return [trainData, trainLabels, testData, testLabels];
  };

  const oneHotEncode = (vocab: string[], value: string) => {
    let oneHot = new Array(vocab.length).fill(0);
    oneHot[vocab.indexOf(value)] = 1;
    return oneHot;
  };

  main = async () => {
    const STEP_SIZE = 1;

    step_count = 0;
    let W = await nv.random([4, 8]);
    W = await W.mul(0.01);
    let b = await nv.zeros([1, 8]);
    let W2 = await nv.random([8, 3]);
    W2 = await W2.mul(0.01);
    let b2 = await nv.zeros([1, 3]);

    let [trainData, trainLabels, testData, testLabels] = await prepareData();

    let vocab = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"];

    trainLabels = trainLabels.map((t: any) => oneHotEncode(vocab, t));
    testLabels = testLabels.map((t: any) => oneHotEncode(vocab, t));

    let trainDataTensor = nv.tensor(trainData as number[][]);
    let trainLabelsTensor = nv.tensor(trainLabels as number[][]);
    let testDataTensor = nv.tensor(testData as number[][]);
    let testLabelsTensor = nv.tensor(testLabels as number[][]);
    let train_samples_n = (await trainDataTensor.shape())[0];


    const fnnParams: fnnParams = {
      layers: [4, 8, 3],
      activation: "reLU",
      LR: LR,
      stepSize: STEP_SIZE,
      batchSize: 1,
      epochs: training_steps,
      logEvery: 10,
    }

    await nv.fnn(
      [trainDataTensor, trainLabelsTensor],
      [testDataTensor, testLabelsTensor],
      fnnParams
    )

    const evaluate = async (X: Tensor, y: Tensor): Promise<number> => {
      let hidden_layer_eval = await (await X.dot(W)).add(b);
      let scores_eval = await (await hidden_layer_eval.dot(W2)).add(b2);
      let predicted_class = await scores_eval.argmax(1);

      let y_argmax = await y.argmax(1);
      let compared = await predicted_class.compare(y_argmax, 1);
      let correctN = await (await compared.sum()).values();

      return Number(((correctN / y.shape()[0]) * 100).toFixed(2));
    };

    init_train_acc = (await evaluate(trainDataTensor, trainLabelsTensor)) + "%";
    final_train_acc = "";

    init_test_acc = (await evaluate(testDataTensor, testLabelsTensor)) + "%";
    final_test_acc = "";

    const fit = async () => {
      while (step_count < training_steps && !stop_signal) {
        // console.log(step_count);
        // https://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html
        let t1 = performance.now();

        let hidden_layer = await (
          await (await trainDataTensor.dot(W)).add(b)
        ).reLU();
        let scores = await (await hidden_layer.dot(W2)).add(b2);

        let exp_scores = await scores.exp();
        let probs = await exp_scores.div(await exp_scores.sum(1));

        let dscores = await probs.minus(trainLabels);
        dscores = await dscores.div(train_samples_n);

        let dW2 = await (await hidden_layer.transpose()).dot(dscores);
        let db2 = await dscores.sum(0);
        let dhidden = await dscores.dot(await W2.transpose());
        dhidden = await dhidden.gradientReLU(hidden_layer);

        let dW = await (await trainDataTensor.transpose()).dot(dhidden);
        let db = await dhidden.sum(0);

        dW2 = await dW2.add(await W2.mul(LR));
        dW = await dW.add(await W.mul(LR));

        W = await W.add(await dW.mul(-STEP_SIZE));
        b = await b.add(await db.mul(-STEP_SIZE));
        W2 = await W2.add(await dW2.mul(-STEP_SIZE));
        b2 = await b2.add(await db2.mul(-STEP_SIZE));

        if (step_count % 20 == 0) {
          console.log(await evaluate(trainDataTensor, trainLabelsTensor));
        }

        let t10 = performance.now();

         console.log(step_count, t10-t1)

          step_count++;
      }
    };

    //await fit();

    fit_btn_text = "▶";
    stop_signal = true;
    final_train_acc =
      (await evaluate(trainDataTensor, trainLabelsTensor)) + "%";
    final_test_acc = (await evaluate(testDataTensor, testLabelsTensor)) + "%";
  };

  if (browser) {
    (async () => {
      await nv.init({ backend });
    })();
  }
</script>

<head>
  <title>Iris Classifier</title>
</head>

<body class="text-slate-700 ">
  <nav class="mt-6 ml-6 mb-6">
    <a class="text-3xl text-red-600" href="/">Demos</a>
  </nav>

  <div class="max-w-3xl mx-auto mb-12">
    <h1 class="text-2xl ">Iris Dataset Classification</h1>

    <label for="backend-select ">Backend: </label>
    <select
      name="backend"
      id="backend-select"
      bind:value={backend}
      class="bg-stone-100 rounded p-2 shadow"
    >
      <option value="auto">Auto</option>
      <option value="js">JS</option>
      <option value="webgpu" disabled={!nv.webgpuAvailable()}>WebGPU</option>
    </select>
    <h1 class="text-lg ">Linear Classifier</h1>
    <label for="training_steps"># Training Steps:</label>

    <input
      type="number"
      min="0"
      name=""
      id="training_steps"
      class="ring-2 rounded"
      bind:value={training_steps}
    />
    <label for="learning_rate">Learning Rate (0-1):</label>

    <input
      type="number"
      step="0.00001"
      min="0"
      max="1"
      name=""
      class="ring-2 rounded"
      bind:value={LR}
    />
    <button
      class="rounded-full bg-red-500 text-white w-12 h-12 hover:bg-red-600 active:bg-red-700"
      on:click={() => {
        if (stop_signal) {
          stop_signal = false;
          main();
          fit_btn_text = "■";
        } else {
          fit_btn_text = "▶";
          stop_signal = true;
        }
      }}>{fit_btn_text}</button
    >
    <p id="step_count">Step #: {step_count}</p>

    <table class="text-center">
      <thead class="border-b border-slate-700">
        <th class="font-normal" />
        <th class="font-normal w-24">Initial</th>
        <th class="font-normal w-24">Final</th>
      </thead>
      <tbody>
        <tr class="border-b border-slate-700">
          <th class="font-normal">Train Accuracy</th>
          <td>{init_train_acc}</td>
          <td>{final_train_acc}</td>
        </tr>
        <tr>
          <th class="font-normal">Test Accuracy</th>
          <td>{init_test_acc}</td>
          <td>{final_test_acc}</td>
        </tr>
      </tbody>
    </table>

    <div
      id="canvas"
      class="w-fit mx-auto overflow-clip rounded-md
         shadow-md"
    />

    <div class="mt-8 indent-8 space-y-4">
      <p>
        This webpage trains a neural network with a hidden layer reLU
        activation, within your web browser to predict the species of an Iris
        flower based on its input variables. The Iris flower dataset contains
        the input variables (features) of Sepal Length, Sepal Width, Petal
        Length, and Petal Width for 150 different Iris flowers. These flowers
        belong to three different species: Setosa, Versicolor, and Virginica.
      </p>
    </div>
  </div>
</body>
