<script lang="ts">
    import { browser } from "$app/environment";
    import pako from "pako";
    import p5 from "p5";

    import * as nv from "nervous";
    import type { Tensor } from "nervous/types/tensor";

    let loading_msgs = []

    if (browser) {

        const oldCanvases = document.getElementsByClassName("p5Canvas");
        while (oldCanvases.length > 0) {
            oldCanvases[0].remove();
        }

        const main = async () => {
            await nv.init();
            let trainPixels, trainLabels, testPixels, testLabels;
            const loadFiles = async () => {
                let dataFileBuffer, labelFileBuffer;
                const fromHexString = (hexString: string) =>
                    Uint8Array.from(
                        hexString
                            .match(/.{1,2}/g)
                            .map((byte) => parseInt(byte, 16))
                    );

                // loading_msgs = [...loading_msgs, "Loading training set files..."]
                // loading_msgs = [...loading_msgs, "Loading 60k images..."]
                // dataFileBuffer = pako.ungzip(
                //     fromHexString(
                //         (await import("./train-images-idx3-ubyte.gz.data?hex"))
                //             .default
                //     )
                // );
                // loading_msgs = [...loading_msgs, "Loading 60k labels..."]
                //
                // labelFileBuffer = pako.ungzip(
                //     fromHexString(
                //         (await import("./train-labels-idx1-ubyte.gz.data?hex"))
                //             .default
                //     )
                // );
                //
                // loading_msgs = [...loading_msgs, "Processing files..."]
                // trainPixels = new Array(60000);
                // trainLabels = new Array(60000);
                //
                // for (let image = 0; image < 60000; image++) {
                //     let pixelCols = new Array(28);
                //     for (let y = 0; y < 28; y++) {
                //         let pixelRows = new Array(28);
                //         pixelCols[y] = pixelRows;
                //         for (let x = 0; x < 28; x++) {
                //             pixelCols[y][x] =
                //                 dataFileBuffer[image * 784 + (x + y * 28) + 16];
                //         }
                //     }
                //     trainPixels[image] = pixels;
                //     trainLabels[image] = labelFileBuffer[image + 8];
                // }


                loading_msgs = [...loading_msgs, "Loading testing set files..."]
                loading_msgs = [...loading_msgs, "Loading 10k images..."]
                dataFileBuffer = pako.ungzip(
                    fromHexString(
                        (await import("./t10k-images-idx3-ubyte.gz.data?hex"))
                            .default
                    )
                );
                loading_msgs = [...loading_msgs, "Loading 10k labels..."]
                labelFileBuffer = pako.ungzip(
                    fromHexString(
                        (await import("./t10k-labels-idx1-ubyte.gz.data?hex"))
                            .default
                    )
                );

                loading_msgs = [...loading_msgs, "Processing files..."]
                testPixels = new Array(10000);
                testLabels = new Array(10000);

                for (let image = 0; image < 10000; image++) {
                    let pixelCols = new Array(28);
                    for (let y = 0; y < 28; y++) {
                        let pixelRows = new Array(28);
                        pixelCols[y] = pixelRows;
                        for (let x = 0; x < 28; x++) {
                            pixelCols[y][x] =
                                dataFileBuffer[image * 784 + (x + y * 28) + 16];
                        }
                    }
                    testPixels[image] = pixelCols;
                    testLabels[image] = labelFileBuffer[image + 8];
                }
                loading_msgs.push("Done!")
            };
            await loadFiles();


            const s = (p) => {
                let gp;
                p.setup = async () => {
                    p.createCanvas(1120, 1000);
                    gp = p.createGraphics(p.width, p.height);
                    gp.pixelDensity(1);
                    p.noLoop();
                };

                p.draw = function () {
                    p.clear(0, 0, 0, 0);

                    for (let image = 0; image <= 10; image++) {
                        gp.reset();
                        gp.loadPixels();
                        for (let x = 0; x < 28; x++) {
                            for (let y = 0; y < 28; y++) {
                                let index = (p.width * y + x) * 4;
                                let shade = testPixels[image][y][x]
                                gp.pixels[index + 0] = shade;
                                gp.pixels[index + 1] = shade;
                                gp.pixels[index + 2] = shade;
                                gp.pixels[index + 3] = 255;
                            }
                        }
                        gp.updatePixels();
                        p.image(
                            gp,
                            Math.floor(image % 40) * 28,
                            Math.floor(image / 40) * 50,
                            p.width,
                            p.height
                        );
                        p.fill(160, 160, 160)
                        p.text(
                            testLabels[image],
                            9 + Math.floor(image % 40) * 28,
                            Math.floor(image / 40) * 50 + 40
                        );
                    }
                };
            };

            new p5(s);


            let layer_dims = [784, 16, 10];
            let testTensor = nv.tensor(testPixels);
            let testLabelsTensor = nv.tensor(testLabels);
        };
        main();
    }
</script>

<head>
    <title>MNIST Digit Recognition</title>
</head>

{#each loading_msgs as msg}
    <p id="loading_msg">{msg}</p>
{/each}
