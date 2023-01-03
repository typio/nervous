<script lang="ts">
    import nv from "nervous";
    import P5 from "p5-svelte";
    import type { Sketch } from "p5-svelte";
    import Katex from "$lib/components/Katex.svelte";

    import iris_raw_data from "./iris.data?raw";
    import { browser } from "$app/environment";
    import type { Tensor } from "nervous/types/tensor";

    let prettyData = "";
    let stop_signal = true;
    let main = () => {};
    let step_count = 0;
    let init_train_acc = "";
    let final_train_acc = "";
    let init_test_acc = "";
    let final_test_acc = "";
    let fit_btn_text = "▶";

    let weight_vals: number[][] = [[0]];
    let bias_vals: number[] = [];
    let output_vals: number[] = [];
    let accuracies: number[] = [];

    let training_steps = 250;
    let LR = 0.001;

    let sketch: Sketch = () => {};

    if (browser) {
        (async () => {
            await nv.init({ backend: "js" });

            let data = iris_raw_data.split("\n");
            data = data.filter((row) => (row === "" ? false : true)); // remove empty lines

            prettyData = " SL\t SW\t PL\t PW\t Species\n";
            for (let i = 0; i < data.length; i++) prettyData += data[i] + "\n";

            const prepareData = async () => {
                data = iris_raw_data.split("\n");
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

            const forward = async (
                input: Tensor,
                W: Tensor,
                b: Tensor
            ): Promise<Tensor> => {
                let output = await (await input.matmul(W)).add(b, 1);
                return output;
            };

            const oneHotEncode = (vocab: string[], value: string) => {
                let oneHot = new Array(vocab.length).fill(0);
                oneHot[vocab.indexOf(value)] = 1;

                return oneHot;
            };

            const evaluate = async (
                inputData: Tensor,
                inputLabels: Tensor,
                W: Tensor,
                b: Tensor
            ): Promise<[Tensor, number]> => {
                let output = await forward(inputData, W, b);

                let outputArr = output.values() as number[];
                let correctOutputArr = inputLabels.values() as number[];

                let correctN = 0;
                for (let i = 0; i < outputArr.length; i++) {
                    if (
                        nv.tensor(outputArr[i]).argmax() ===
                        nv.tensor(correctOutputArr[i]).argmax()
                    )
                        correctN++;
                }

                return [
                    output,
                    Number(((correctN / outputArr.length) * 100).toFixed(2)),
                ];
            };

            weight_vals = new Array(4).fill(new Array(3).fill(0));
            bias_vals = new Array(3).fill(0);
            output_vals = new Array(3).fill(0);

            main = async () => {
                accuracies = [];

                const STEP_SIZE = 1;

                step_count = 0;
                let W = nv.randomNormal([4, 3]);
                let b = nv.zeros(3);

                let [trainData, trainLabels, testData, testLabels] =
                    await prepareData();

                let vocab = [
                    "Iris-versicolor",
                    "Iris-virginica",
                    "Iris-setosa",
                ]; // fixed order so I can label on diagram

                trainLabels = trainLabels.map((t: any) =>
                    oneHotEncode(vocab, t)
                );
                testLabels = testLabels.map((t: any) => oneHotEncode(vocab, t));

                let trainDataTensor = nv.tensor(trainData as number[][]);
                let trainLabelsTensor = nv.tensor(trainLabels as number[][]);
                let testDataTensor = nv.tensor(testData as number[][]);
                let testLabelsTensor = nv.tensor(testLabels as number[][]);
                let train_samples_n = trainDataTensor.shape()[0];

                init_train_acc =
                    (
                        await evaluate(trainDataTensor, trainLabelsTensor, W, b)
                    )[1] + "%";
                final_train_acc = "";

                init_test_acc =
                    (
                        await evaluate(testDataTensor, testLabelsTensor, W, b)
                    )[1] + "%";
                final_test_acc = "";

                weight_vals = W.values() as unknown as number[][];
                bias_vals = b.flatValues();
                const fit = async () => {
                    let [output, accuracy] = await evaluate(
                        trainDataTensor,
                        trainLabelsTensor,
                        W,
                        b
                    );

                    accuracies.push(accuracy);

                    let probs = output.softmax();

                    let correct_logprobs = probs
                        .mul(trainLabelsTensor)
                        .sum(1)
                        .log()
                        .mul(-1);

                    let data_loss =
                        correct_logprobs.sum().values() / train_samples_n;
                    let reg_loss = 0.5 * LR * W.mul(W).sum().values();
                    let loss = data_loss + reg_loss;

                    step_count++;

                    let dOutput = probs.minus(trainLabelsTensor);

                    dOutput = dOutput.div(train_samples_n);

                    let dW = trainDataTensor.transpose().matmul(dOutput);

                    let db = dOutput.sum(0);

                    dW = dW.add(W.mul(LR));

                    // perform a parameter update
                    W = W.add(dW.mul(-1 * STEP_SIZE));
                    b = b.add(db.mul(-1 * STEP_SIZE));

                    weight_vals = W.values(3) as unknown as number[][];
                    bias_vals = b.flatValues(3);
                    output_vals = output.flatValues(3);

                    if (step_count < training_steps && !stop_signal) {
                        requestAnimationFrame(fit);
                    } else {
                        fit_btn_text = "▶";
                        stop_signal = true;
                        final_train_acc =
                            (
                                await evaluate(
                                    trainDataTensor,
                                    trainLabelsTensor,
                                    W,
                                    b
                                )
                            )[1] + "%";
                        final_test_acc =
                            (
                                await evaluate(
                                    testDataTensor,
                                    testLabelsTensor,
                                    W,
                                    b
                                )
                            )[1] + "%";
                    }
                };
                requestAnimationFrame(fit);
            };
        })();

        sketch = (p) => {
            p.setup = () => {
                p.createCanvas(400, 400);
                p.textAlign(p.CENTER, p.CENTER);
            };

            p.draw = () => {
                p.clear(0, 0, 0, 0);
                p.background(244, 244, 255);

                p.fill(p.color("black"));

                p.text("SL", 50, 50);
                p.text("SW", 50, 150);
                p.text("PL", 50, 250);
                p.text("PW", 50, 350);

                p.text("W", 140, 40);
                p.text("b", 270, 70);
                p.text("y", 300, 65);

                p.text("Versicolor", 360, 100);
                p.text("Virginica", 360, 200);
                p.text("Setosa", 360, 300);

                // first layer
                for (let i = 0; i < 4; i++) {
                    // lines
                    for (let j = 0; j < 3; j++) {
                        p.line(100, 50 + i * 100, 300, 100 + j * 100);

                        let t = 0.2;
                        let y_offset = 5; //j % 2 == 0 ? 5 : -5
                        let m =
                            (100 + j * 100 - (50 + i * 100) + y_offset) / 200;
                        let midX = (1 - t) * 100 + t * 300;
                        let midY =
                            (1 - t) * (50 + i * 100) +
                            t * (100 + j * 100) -
                            y_offset;

                        p.fill(p.color("red"));
                        p.textStyle(p.BOLD);
                        if (weight_vals.length == 4) {
                            if (weight_vals[0].length == 3) {
                                p.translate(midX, midY);
                                p.rotate(Math.atan(m));
                                p.text(weight_vals[i][j], 0, 0);
                                p.rotate(-Math.atan(m));
                                p.translate(-midX, -midY);
                            }
                        }
                        p.fill(p.color("white"));

                        p.ellipse(100, 50 + i * 100, 40);
                    }

                    // second layer
                    for (let i = 0; i < 3; i++) {
                        p.fill(p.color("white"));
                        p.ellipse(300, 100 + i * 100, 40);
                        if (bias_vals.length === 3) {
                            // console.log(bias_vals[i]);
                            p.fill(p.color("red"));
                            p.text(bias_vals[i], 300, 100 + i * 100);
                        }
                    }
                }

                // acc graph
                p.strokeWeight(0);
                p.fill(p.color("white"));
                p.rect(280, 5, 120, 50);
                p.fill(p.color("limegreen"));
                for (let i = 0; i < accuracies.length; i++) {
                    p.ellipse(
                        280 + (i / accuracies.length) * 120,
                        accuracies[i] * -0.45 + 50,
                        2
                    );
                }
                p.strokeWeight(1);
                p.fill(p.color("dodgerblue"));
                p.text("Accuracy", 350, 65);
                p.text(0, 270, 50);
                p.text(1, 270, 5);
            };
        };
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
        <p>Step #: {step_count}</p>

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
        >
            <P5 {sketch} />
        </div>

        <div class="mt-8 indent-8 space-y-4">
            <p>
                This webpage trains a neural network, specifically a linear
                classifier, within your web browser to predict the species of an
                Iris flower based on its input variables. The Iris flower
                dataset contains the input variables (features) of Sepal Length,
                Sepal Width, Petal Length, and Petal Width for 150 different
                Iris flowers. These flowers belong to three different species:
                Setosa, Versicolor, and Virginica.
            </p>
            <p>
                In the linear classifier, the input variables <Katex math='(x)'/> are multiplied
                by a matrix of weights <Katex math='(W)'/>, and a vector <Katex math='(b)'/> is added to the
                product. This produces a set of weighted input values that
                represent the relative importance of each input variable in
                predicting the species of the flower. The index of the largest
                value in the resultant vector is the predicted species of the
                flower <Katex math='(y)'/>. Initially, <Katex math='W'/> and <Katex math='b'/> are assigned random values, but
                the neural network learns by adjusting these values in a way
                that minimizes future error.
            </p>
            <p>
                Error is calculated using the softmax and cross-entropy
                functions. The softmax function assigns a probability to each
                possible output value (i.e. each species of Iris flower), and
                the cross-entropy function measures the difference between the
                predicted probabilities and the true probabilities (i.e. the
                actual species of the flower). By minimizing this difference,
                the neural network improves its accuracy in correctly predicting
                the species of an Iris flower from its input variables.
            </p>

            <p>Here is the full dataset:</p>
        </div>

        <div class="h-64 overflow-scroll mx-auto w-fit">
            <pre id="iris_data">{prettyData}</pre>
        </div>
    </div>
</body>
