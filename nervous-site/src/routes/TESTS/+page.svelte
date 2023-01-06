<script lang="ts">
    import * as PIXI from "pixi.js";
    import nv from "nervous";
    import { browser } from "$app/environment";

    import tests from "./tests";

    let backend = "auto";

    let testResults: any[] = [];

    const runTests = async () => {
        nv.init({ backend });

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
                        return JSON.stringify(await test.code());
                    } catch (e) {
                        codeErrored = true;
                        return e;
                    }
                })();

                let expectsErrored = false;
                const expectsRes = await (async () => {
                    try {
                        return JSON.stringify(await test.expects());
                    } catch (e) {
                        expectsErrored = true;
                        return e;
                    }
                })();

                testResults[i].results[j] = {
                    ...testResults[i].results[j],
                    res: codeRes === expectsRes,
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
        // runTests();
        const main = async () => {
            await nv.init({ backend: "webgpu" });
            // let n = await nv.tensor([
            //     [0, 0, 0, 0],
            //     [0, 0, 0, 0],
            //     [0, 0, 0, 0],
            // ]);

            // // console.log(n.shape());

            // console.log(JSON.stringify(await n.values()));

            // let m = await nv.tensor([[1], [4], [5]]);
            // let m = await nv.tensor([
            //     [1, 2, 3, 4],
            //     [5, 6, 7, 8],
            //     [9, 10, 11, 12],
            // ]);
            // let m = await nv.tensor([1, 2, 3, 4]);
            // let m = await nv.scalar(69);
            // console.log(m.shape());
            // console.log(await m.values());

            // let res = await (await n.add(m)).toJS();
            // console.log(res.data);

            // console.log(JSON.stringify(await res.values()));
            let a = nv.tensor([1, 2]);
            let b = nv.tensor([4, 4]);
            let result = await a.add(b);

            console.log(await result.values());
             

            let start = performance.now();
            const [n, m] = await Promise.all([
                nv.random([2047, 2047]),
                nv.random([2047, 2047]),
            ]);
            // const n = await nv.random([2047, 2047])
            // const m = await nv.random([2047, 2047])
            let end = performance.now();
            console.log(
                `%c${end - start}ms`,
                "background: dodgerblue; color: white; padding: 3px 4px; border-radius:2px;"
            );

            let res = await n.minus(m);
            console.log(res);

            console.log(await res.values());
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
                    class="px-2 max-h-28 overflow-scroll py-1 ml-4 text-sm bg-slate-900 text-slate-300">{result.code}</pre>
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
