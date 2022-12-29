<script lang="ts">
    import P5 from "p5-svelte";
    // import * as d3 from "d3";
    import nv from "nervous";

    import iris_raw_data from "./iris.data?raw";
    import { browser } from "$app/environment";

    // console.log(iris_raw_data);
//     let sketch;

//     if (browser) {
//         let data = iris_raw_data.split("\n");
//         let prettyData = " SL\t SW\t PL\t PW\t Species\n";
//         for (let i = 0; i < data.length; i++) {
//             prettyData += data[i] + "\n";
//         }
//         // prettyData.unshift(['pl','pw','sl','sw','Species'])
//         document.getElementById("iris_data").innerText = prettyData;

//         const prepareData = async () => {
//             // const text = await d3.text("./iris.data");
//             let data = iris_raw_data.split("\n"); // d3.csvParseRows(text).slice(0, -1); // has a weird 0th element
//             data = data.sort(() => Math.random() - 0.5); // shuffle

//             data = data.map((row) => row.map((s, i) => (i === 4 ? s : +s)));

//             const inputs = data.map((row) => row.slice(0, -1));
//             const targets = [].concat(...data.map((row) => row.slice(-1)));

//             let cutoffIdx = Math.floor(data.length * 0.8);
//             let trainData = inputs.slice(0, cutoffIdx);
//             let trainLabels = targets.slice(0, cutoffIdx);
//             let testData = inputs.slice(cutoffIdx);
//             let testLabels = targets.slice(cutoffIdx);
//             return [trainData, trainLabels, testData, testLabels];
//         };

//         const forward = (input: nv.Tensor, W: nv.Tensor, b: nv.Tensor) => {
//             let output = nv.add(nv.matmul(input, W), b, 1); //input.matmul(W).add(b, 1);
//             return output;
//         };

//         const oneHotEncode = (vocab, value) => {
//             let oneHot = new Array(vocab.length).fill(0);
//             oneHot[vocab.indexOf(value)] = 1;

//             return oneHot;
//         };

//         const evaluate = (
//             inputData: nv.Tensor,
//             inputLabels: nv.Tensor,
//             W: nv.Tensor,
//             b: nv.Tensor
//         ) => {
//             let output = forward(inputData, W, b);
//             let outputArr = nv.getValues(output);
//             let correctOutputArr = nv.getValues(inputLabels); // inputLabels.getValues();

//             let correctN = 0;
//             for (let i = 0; i < outputArr.length; i++) {
//                 if (
//                     nv.argmax(nv.tensor(outputArr[i])) ===
//                     nv.argmax(nv.tensor(correctOutputArr[i]))
//                 )
//                     correctN++;
//             }

//             return [
//                 output,
//                 Number(((correctN / outputArr.length) * 100).toFixed(2)),
//             ];
//         };

//         let step_count_el = document.getElementById("step");
//         let fit_btn_el = document.getElementById("fit_btn");

//         let train_steps_input = document.getElementById("training_steps");
//         let training_steps = Number(train_steps_input?.value);
//         train_steps_input.addEventListener("change", () => {
//             training_steps = train_steps_input.value;
//         });

//         const LR = Number(document.getElementById("learning_rate")?.value);

//         let weight_vals = [[0]];
//         let bias_vals = [];
//         let output_vals = [];

//         let stop_signal = true;

//         let accuracies = [];

//         const main = async () => {
//             accuracies = [];
//             let step_count = 0;

//             const STEP_SIZE = 1;

//             let W = nv.randomNormal([4, 3]);
//             let b = nv.zeros(3);

//             let [trainData, trainLabels, testData, testLabels] =
//                 await prepareData();
//             // let vocab = Array.from(new Set(trainLabels.concat(testLabels)))
//             let vocab = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]; // fixed order so I can label on diagram

//             trainLabels = trainLabels.map((t: any) => oneHotEncode(vocab, t));
//             testLabels = testLabels.map((t: any) => oneHotEncode(vocab, t));

//             trainData = nv.tensor(trainData);
//             trainLabels = nv.tensor(trainLabels);
//             testData = nv.tensor(testData);
//             testLabels = nv.tensor(testLabels);
//             let train_samples_n = trainData.shape[0];

//             document.getElementById("init_train_acc").innerHTML = "";
//             document.getElementById("init_test_acc").innerHTML = "";
//             document.getElementById("final_train_acc").innerHTML = "";
//             document.getElementById("final_test_acc").innerHTML = "";

//             document.getElementById("init_train_acc").innerHTML =
//                 evaluate(trainData, trainLabels, W, b)[1] + "%";
//             document.getElementById("init_test_acc").innerHTML =
//                 evaluate(testData, testLabels, W, b)[1] + "%";

//             weight_vals = nv.getValues(W);
//             bias_vals = nv.getValues(b);
//             const fit = () => {
//                 let [output, accuracy] = evaluate(trainData, trainLabels, W, b);
//                 accuracies.push(accuracy);

//                 let probs = nv.softmax(output) 

//                 let correct_logprobs = probs
//                     .mul(trainLabels)
//                     .sum(1)
//                     .log()
//                     .mul(-1);

//                 let data_loss =
//                     correct_logprobs.sum().getValues() / train_samples_n;
//                 let reg_loss = 0.5 * LR * W.mul(W).sum().getValues();
//                 let loss = data_loss + reg_loss;

//                 step_count++;

//                 let dOutput = probs.minus(trainLabels);
//                 dOutput = dOutput.div(train_samples_n);

//                 let dW = nv.matmul(nv.transpose(trainData), dOutput); // trainData.transpose().matmul(dOutput);
//                 let db = dOutput.sum(0);

//                 dW = dW.add(W.mul(LR));

//                 // perform a parameter update
//                 W = W.add(dW.mul(-1 * STEP_SIZE));
//                 b = b.add(db.mul(-1 * STEP_SIZE));

//                 weight_vals = W.getValues(3);
//                 bias_vals = b.getValues(3);
//                 output_vals = output.getValues(3);

//                 if (step_count <= training_steps && !stop_signal) {
//                     step_count_el.innerHTML = step_count;
//                     requestAnimationFrame(fit);
//                 } else {
//                     fit_btn_el.innerText = "▶";
//                     stop_signal = true;
//                     document.getElementById("final_train_acc").innerHTML =
//                         evaluate(trainData, trainLabels, W, b)[1] + "%";
//                     document.getElementById("final_test_acc").innerHTML =
//                         evaluate(testData, testLabels, W, b)[1] + "%";
//                 }
//             };
//             requestAnimationFrame(fit);
//         };

//         fit_btn_el?.addEventListener("click", () => {
//             if (stop_signal) {
//                 stop_signal = false;
//                 main();
//                 fit_btn_el.innerText = "■";
//             } else {
//                 stop_signal = true;
//             }
//         });

//         sketch = (p) => {
//             p.setup = () => {
//                 p.createCanvas(400, 400);
//                 p.textAlign(p.CENTER, p.CENTER);
//             };

//             p.draw = () => {
//                 p.clear(0);
//                 p.background(244, 244, 255);

//                 p.fill(p.color("black"));
//                 p.text("x", 100, 15);

//                 p.text("SL", 50, 50);
//                 p.text("SW", 50, 150);
//                 p.text("PL", 50, 250);
//                 p.text("PW", 50, 350);

//                 p.text("W", 140, 40);
//                 p.text("b", 270, 70);
//                 p.text("y", 300, 65);

//                 p.text("Versicolor", 360, 100);
//                 p.text("Virginica", 360, 200);
//                 p.text("Setosa", 360, 300);

//                 // first layer
//                 for (let i = 0; i < 4; i++) {
//                     // lines
//                     for (let j = 0; j < 3; j++) {
//                         p.line(100, 50 + i * 100, 300, 100 + j * 100);

//                         let t = 0.2;
//                         let y_offset = 5; //j % 2 == 0 ? 5 : -5
//                         let m =
//                             (100 + j * 100 - (50 + i * 100) + y_offset) / 200;
//                         let midX = (1 - t) * 100 + t * 300;
//                         let midY =
//                             (1 - t) * (50 + i * 100) +
//                             t * (100 + j * 100) -
//                             y_offset;

//                         p.fill(p.color("red"));
//                         p.textStyle(p.BOLD);
//                         if (weight_vals.length == 4) {
//                             if (weight_vals[0].length == 3) {
//                                 p.translate(midX, midY);
//                                 p.rotate(Math.atan(m));
//                                 p.text(weight_vals[i][j], 0, 0);
//                                 p.rotate(-Math.atan(m));
//                                 p.translate(-midX, -midY);
//                             }
//                         }
//                         p.fill(p.color("white"));

//                         p.ellipse(100, 50 + i * 100, 40);
//                     }

//                     // second layer
//                     for (let i = 0; i < 3; i++) {
//                         p.fill(p.color("white"));
//                         p.ellipse(300, 100 + i * 100, 40);
//                         if (bias_vals.length === 3) {
//                             // console.log(bias_vals[i]);
//                             p.fill(p.color("red"));
//                             p.text(bias_vals[i], 300, 100 + i * 100);
//                         }
//                     }
//                 }

//                 // acc graph
//                 p.strokeWeight(0);
//                 p.fill(p.color("white"));
//                 p.rect(280, 5, 120, 50);
//                 p.fill(p.color("limegreen"));
//                 for (let i = 0; i < accuracies.length; i++) {
//                     p.ellipse(
//                         280 + (i / accuracies.length) * 120,
//                         accuracies[i] * -0.45 + 50,
//                         2
//                     );
//                 }
//                 p.strokeWeight(1);
//                 p.fill(p.color("dodgerblue"));
//                 p.text("Accuracy", 350, 65);
//                 p.text(0, 270, 50);
//                 p.text(1, 270, 5);
//             };
//         };
//     }
//     // new P5(s, document.getElementById("P5CanvasParent") ?? undefined);
</script>

