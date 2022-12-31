<script lang="ts">
    import * as PIXI from "pixi.js";
    import nv from "nervous";
    import { browser } from "$app/environment";

    import tests from "./tests";

    let backend = "auto";

    let testResults: any[] = [];

    const runTests = async () => {
        await nv.init({ backend });

        testResults = [];
        for (let i = 0; i < tests.length; i++) {
            let suite = tests[i];
            testResults.push({
                name: suite.suite,
                results: [],
            });

            for (let j = 0; j < suite.tests.length; j++) {
                // testResults.suite.push(test.code());
                let test = suite.tests[j];
                testResults[i].results.push({
                    name: test.name,
                    res: null,
                });
                testResults = testResults;

                let codeErrored = false;
                const codeRes = await (async () => {
                    try {
                        return await test.code();
                    } catch (e) {
                        codeErrored = true;
                        return e;
                    }
                })();

                let expectsErrored = false;
                const expectsRes = await (async () => {
                    try {
                        return await test.expects();
                    } catch (e) {
                        expectsErrored = true;
                        return e;
                    }
                })();

                testResults[i].results[j] = {
                    ...testResults[i].results[j],
                    res: JSON.stringify(codeRes) === JSON.stringify(expectsRes),
                    code: test.code,
                    expects: test.expects,
                    codeErrored,
                    expectsErrored,
                    codeRes,
                    expectsRes,
                };
            }
        }
    };

    if (browser) {
        const main = async () => {
            {
                await nv.init();
                const size = 512;
                let result = await nv.random([size, size]);
                let resultValues = result.values();

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
                        else graphics.beginFill(0xff0000 * resultValues[i][j]);

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
    <title>Tests</title>
</head>

<nav class="mt-6 ml-6">
    <a class="text-3xl text-red-600" href="/">Demos</a>
</nav>

<label for="backend-select">Backend: </label>
<select name="backend" id="backend-select" bind:value={backend}>
    <option value="auto">Auto</option>
    <option value="js">JS</option>
    <option value="webgpu" disabled={!nv.webgpuAvailable()}>WebGPU</option>
</select>

<button
    class="rounded shadow text-slate-100 bg-green-500 active:bg-green-600 p-2"
    on:click={runTests}
>
    Run Tests
</button>

{#if testResults.length !== 0}
    <button
        class="rounded shadow text-slate-100 bg-red-500 active:bg-red-600 p-2"
        on:click={() => {
            testResults = [];
        }}
    >
        Clear Results
    </button>
{/if}

{#each testResults as suite}
    {#if suite}
        <h3 class="text-xl font-medium">{suite.name}</h3>
        {#each suite.results || [] as result}
            <div class="inline-flex">
                <p>{result.name}</p>
                <pre
                    class="px-2 py-1 ml-4 text-sm bg-slate-900 text-slate-300">{result.code}</pre>
            </div>
            <div>
                {#if result.res === null}
                    <p>running</p>
                {:else}
                    <!-- <p>{result.res}</p> -->

                    {#if result.codeErrored || result.expectsErrored}
                        <div class="bg-red-300">
                            {#if result.codeErrored}
                                ❌ Test {result.codeRes}
                            {/if}
                        </div>
                        <div class="bg-red-300">
                            {#if result.expectsErrored}
                                ❌ Expected Result Calculation {result.expectsRes}
                            {/if}
                        </div>
                    {:else if result.res}
                        <div class="bg-green-300">
                            ✅ {result.codeRes} ≡ {result.expectsRes}
                        </div>
                    {:else}
                        <div class="bg-red-300">
                            ❌ {result.codeRes} ≢ {result.expectsRes}
                        </div>
                    {/if}
                {/if}
            </div>
        {/each}
    {/if}
{/each}

<p>is this random enough?</p>
