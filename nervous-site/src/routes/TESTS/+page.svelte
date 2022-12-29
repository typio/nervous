<script lang="ts">
    import * as PIXI from "pixi.js";
    import nv from "nervous";
    import { browser } from "$app/environment";

    if (browser) {
        const main = async () => {
            {
                await nv.init({ backend: "webgpu" });
                const size = 512;
                let result = await nv.random([size, size], 42);
                let resultValues = result.values();
                console.log(result.flatValues(3));

                const app = new PIXI.Application({
                    width: 512,
                    height: 512,
                    antialias: true,
                });

                document.body.appendChild(app.view);
                const graphics = new PIXI.Graphics();
                for (let i = 0; i < result.shape()[0]; i++) {
                    for (let j = 0; j < result.shape()[1]; j++) {
                        if (i % 64 === 0 || j % 64 === 0)
                            graphics.beginFill(0x000000);
                        else graphics.beginFill(0xffffff * resultValues[i][j]);

                        graphics.drawRect(i, j, 1, 1);
                        graphics.endFill();
                    }
                }
                app.stage.addChild(graphics);
            }

            {
                await nv.init({ backend: "webgpu" });
                const size = 512;
                console.log(
                    `%cspeed test of random([${size}, ${size}])`,
                    "background: #00ff00;"
                );
                let start = performance.now();

                let tensor1 = await nv.random([size, size], 42);
                let tensor2 = await nv.random([size, size], 69);
                let mid = performance.now();

                {
                    let result = await tensor1.matmul(tensor2);
                    console.log(result);
                }
                console.log(
                    `%cwebgpu time: ${mid - start}ms`,
                    "background: #ffff00;"
                );

                await nv.init({ backend: "js" });
                mid = performance.now();

                tensor1 = await nv.random([size, size]);
                tensor2 = await nv.random([size, size]);
                let end = performance.now();

                {
                    let result = await tensor1.matmul(tensor2);
                    console.log(result);
                }

                console.log(
                    `%cjs time: ${end - mid}ms`,
                    "background: #ffff00;"
                );
            }
        };
        main();
    }
</script>

<head>
    <meta content="text/html;charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="shortcut icon" href="stolen-favicon.png" type="image/png" />
    <title>Tests</title>
</head>

<nav class='mt-6 ml-6'>
    <a class="text-3xl text-red-600" href='/'>Demos</a>
</nav>

is this random enough?
