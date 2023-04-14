<script lang="ts">
    import * as nv from "nervous";
    import Katex from "$lib/components/Katex.svelte";
    import P5 from "p5-svelte";
    import type { Sketch } from "p5-svelte";

    import iris_raw_data from "./iris.data?raw";
    import { browser } from "$app/environment";
    import type { Tensor } from "nervous/types/tensor";

    import iris_image from "./iris-images.png";

    let stop_signal = true;
    let main = () => {};
    let step_count = 0;
    let init_train_acc = "";
    let final_train_acc = "";
    let init_test_acc = "";
    let final_test_acc = "";
    let fit_btn_text = "▶";


    let training_steps = 200;
    let LR = 0.001;
    let log_step_count = 20

    let predictSepalLength = 6.1;
    let predictSepalWidth = 2.8;
    let predictPetalLength = 4;
    let predictPetalWidth = 1.3;
    let predictResult = "";
    const vocab = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"];
    let sketch: Sketch = () => {};

    const prepareData = async () => {
        let data = iris_raw_data.split("\n");
        data = data.filter((row) => (row === "" ? false : true)); // remove empty lines
        data = data.sort(() => Math.random() - 0.5); // shuffle

        let mixed_type_data = data.map((row) =>
            row.split(",").map((s, i) => (i === 4 ? s : +s))
        );

        const inputs = mixed_type_data.map((row) => row.slice(0, -1));

        const targets = mixed_type_data.map((row) => row.slice(-1)[0]);

        let cutoffIdx = Math.floor(data.length * 0.7);
        let trainData = inputs.slice(0, cutoffIdx);
        let trainLabels = targets.slice(0, cutoffIdx);
        let testData = inputs.slice(cutoffIdx);
        let testLabels = targets.slice(cutoffIdx);

        return [trainData, trainLabels, testData, testLabels];
    };

    let means: number[];
    let stds: number[];
    const normalize = (data: number[][], set?: boolean) => {
        if (set) {
            means = data[0].map((_, i) => {
                return data.reduce((acc, row) => acc + row[i], 0) / data.length;
            });

            stds = data[0].map((_, i) => {
                return Math.sqrt(
                    data.reduce(
                        (acc, row) => acc + Math.pow(row[i] - means[i], 2),
                        0
                    ) / data.length
                );
            });
        }

        return data.map((row) =>
            row.map((value, i) => (value - means[i]) / stds[i])
        );
    };

    const oneHotEncode = (vocab: string[], value: string) => {
        let oneHot = new Array(vocab.length).fill(0);
        oneHot[vocab.indexOf(value)] = 1;
        return oneHot;
    };

    let W, b, W2, b2;

    main = async () => {
        const STEP_SIZE = 1;

        step_count = 0;
        W = await nv.random([4, 8]);
        W = await W.mul(0.01);
        b = await nv.zeros([1, 8]);
        W2 = await nv.random([8, 3]);
        W2 = await W2.mul(0.01);
        b2 = await nv.zeros([1, 3]);

        let [trainData, trainLabels, testData, testLabels] =
            await prepareData();

        // Normalize the input data
        trainData = normalize(trainData, true);
        testData = normalize(testData);

        trainLabels = trainLabels.map((t: any) => oneHotEncode(vocab, t));
        testLabels = testLabels.map((t: any) => oneHotEncode(vocab, t));

        let trainDataTensor = nv.tensor(trainData as number[][]);
        let trainLabelsTensor = nv.tensor(trainLabels as number[][]);
        let testDataTensor = nv.tensor(testData as number[][]);
        let testLabelsTensor = nv.tensor(testLabels as number[][]);
        let train_samples_n = (await trainDataTensor.shape())[0];

        const evaluate = async (X: Tensor, y: Tensor): Promise<number> => {
            let hidden_layer_eval = (await (await X.dot(W)).add(b)).leakyRelu();
            let scores_eval = await (await hidden_layer_eval.dot(W2)).add(b2);
            let predicted_class = await scores_eval.argmax(1);

            let y_argmax = await y.argmax(1);
            let compared = await predicted_class.eq(y_argmax, 1);
            let correctN = await (await compared.sum()).values();

            return Number(((correctN / y.shape()[0]) * 100).toFixed(2));
        };

        init_train_acc =
            (await evaluate(trainDataTensor, trainLabelsTensor)) + "%";
        final_train_acc = "";

        init_test_acc =
            (await evaluate(testDataTensor, testLabelsTensor)) + "%";
        final_test_acc = "";

        const fit = async () => {
            while (step_count < training_steps && !stop_signal) {
                let current_LR = LR * (1 / (1 + step_count * 1e-4));
                // console.log(await evaluate(trainDataTensor, trainLabelsTensor));
                // console.log(step_count);
                // https://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html
                let t1 = performance.now();

                let hidden_layer = await (
                    await (await trainDataTensor.dot(W)).add(b)
                ).leakyRelu();
                let scores = await (await hidden_layer.dot(W2)).add(b2);

                let probs = await scores.softmax(1);

                let dscores = await probs.minus(trainLabels);
                dscores = await dscores.div(train_samples_n);

                let dW2 = await (await hidden_layer.transpose()).dot(dscores);
                let db2 = await dscores.sum(0);
                let dhidden = await dscores.dot(await W2.transpose());
                dhidden = await dhidden.gradientLeakyRelu(hidden_layer);

                let dW = await (await trainDataTensor.transpose()).dot(dhidden);
                let db = await dhidden.sum(0);

                dW2 = await dW2.add(await W2.mul(current_LR));
                dW = await dW.add(await W.mul(current_LR));

                W = await W.add(await dW.mul(-STEP_SIZE));
                b = await b.add(await db.mul(-STEP_SIZE));
                W2 = await W2.add(await dW2.mul(-STEP_SIZE));
                b2 = await b2.add(await db2.mul(-STEP_SIZE));

                if (step_count % log_step_count == 0) {
                    // console.log(
                    //     "acc: ",
                    //     await evaluate(trainDataTensor, trainLabelsTensor),
                    //     "%"
                    // );
                    await (() => new Promise((r) => setTimeout(r, 1)))();
                }

                let t10 = performance.now();

                // console.log(step_count, t10 - t1);

                step_count++;
            }
        };

        await fit();

        fit_btn_text = "▶";
        stop_signal = true;
        final_train_acc =
            (await evaluate(trainDataTensor, trainLabelsTensor)) + "%";
        final_test_acc =
            (await evaluate(testDataTensor, testLabelsTensor)) + "%";
    };

    const predict = async (data: number[]) => {
        if (means == undefined || stds == undefined) {
            console.warn("means or stds are undefined");
            return;
        }
        let input = normalize([data]);
        let inputTensor = nv.tensor(input as number[][]);
        let hidden_layer = inputTensor.dot(W).add(b).relu();
        let scores = hidden_layer.dot(W2).add(b2);
        let probs = scores.softmax(1);
        let predicted_class = probs.argmax(1);

        let predicted_class_value = await predicted_class.values();

        let predicted_class_name = vocab[predicted_class_value];
        predictResult = predicted_class_name;
    };

    if (browser) {
        (async () => {
            await nv.init();
        })();

        sketch = (p) => {
            p.setup = () => {
                p.createCanvas(400, 400);
                // p.textAlign(p.CENTER, p.CENTER);
	            p.rectMode(p.CENTER);
            };

            p.draw = () => {
                p.background(240);
                p.background(60, 230, 90);

                p.translate(p.width / 2, p.height / 2);
                for (let i = 0; i < 3; i++) {
                    p.push();
                    p.rotate(i * 2 * p.TWO_PI / 6);
                    p.noStroke();
                    p.fill(255, 255, 0);
                    p.ellipse(
                        100,
                        0,
                        predictSepalLength * 20,
                        predictSepalWidth * 20
                    );
                    p.rotate(i * 2 + 1 * p.TWO_PI / 6);
                    p.fill(0, 100, 180);
                    p.ellipse(
                        100,
                        0,
                        predictPetalLength * 20,
                        predictPetalWidth * 20
                    );

                    p.pop();
                }
            };
        };
    }
</script>

<head>
    <title>Iris Classifier</title>
</head>

<body class="flex">
    <div class="max-w-3xl mx-auto mb-12">
        <h1 class="text-2xl ">Iris Dataset Classification</h1>

        <h2 class="text-lg mt-8">Linear Classifier</h2>
        <label for="training_steps"># Training Steps:</label>

        <input
            type="number"
            step="1"
            class="border-b-2 border-slate-300 dark:border-slate-500 dark:bg-slate-800 rounded decoration-none
            outline-none text-center w-24"
            min="0"
            name=""
            id="training_steps"
            bind:value={training_steps}
        />
        <label for="learning_rate">Learning Rate (0-1):</label>

        <input
            type="number"
            step="0.00001"
            class="border-b-2 border-slate-300 dark:border-slate-500 dark:bg-slate-800 rounded decoration-none
            outline-none text-center w-24"
            min="0"
            max="1"
            name=""
            bind:value={LR}
        />

        <label for="log_step_count">Log Every: </label>
        <input
            type="number"
            step="1"
            class="border-b-2 border-slate-300 dark:border-slate-500 dark:bg-slate-800 rounded decoration-none
            outline-none text-center w-24"
            min="0"
            name=""
            id="log_step_count"
            bind:value={log_step_count}
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

        <div class="flex flex-col mx-auto mt-8">
            <h2 class="text-lg">Use Model</h2>
            <div class="flex flex-row place-content-around my-4">
                <div class="flex flex-col">
                    <label for="sl">Sepal Length - {predictSepalLength}</label>
                    <input
                        name="sl"
                        type="range"
                        min="4.2"
                        max="8"
                        step="0.1"
                        bind:value={predictSepalLength}
                    />
                </div>

                <div class="flex flex-col">
                    <label for="sw">Sepal Width - {predictSepalWidth}</label>
                    <input
                        name="sw"
                        type="range"
                        min="1.9"
                        max="4.5"
                        step="0.1"
                        bind:value={predictSepalWidth}
                    />
                </div>

                <div class="flex flex-col">
                    <label for="pl">Petal Length - {predictPetalLength}</label>
                    <input
                        name="pl"
                        type="range"
                        min="0.1"
                        max="7"
                        step="0.1"
                        bind:value={predictPetalLength}
                    />
                </div>

                <div class="flex flex-col">
                    <label for="pw">Petal Width - {predictPetalWidth}</label>
                    <input
                        name="pw"
                        type="range"
                        min="0.1"
                        max="2.6"
                        step="0.1"
                        bind:value={predictPetalWidth}
                    />
                </div>
            </div>

            <div
                id="canvas"
                class="w-fit mx-auto overflow-clip rounded-md
         shadow-md mb-4"
            >
                <P5 {sketch} />
            </div>
            <img alt="The project logo" src={iris_image} />

            <div class="flex mx-auto mt-4">
                <button
                    class="rounded shadow bg-green-500 w-20 py-2 text-white "
                    on:click={() => {
                        predict([
                            predictSepalLength,
                            predictSepalWidth,
                            predictPetalLength,
                            predictPetalWidth,
                        ]);
                    }}>Predict</button
                >
                <p class="ml-4 leading-10">Prediction: {predictResult}</p>
            </div>
        </div>

        <div class="mt-8 indent-8 space-y-4">
            <p>
                This webpage trains a neural network with a hidden layer reLU
                activation, within your web browser to predict the species of an
                Iris flower based on its input variables. The Iris flower
                dataset contains the input variables (features) of Sepal Length,
                Sepal Width, Petal Length, and Petal Width for 150 different
                Iris flowers. These flowers belong to three different species:
                Setosa, Versicolor, and Virginica.
            </p>
        </div>
    </div>
</body>
