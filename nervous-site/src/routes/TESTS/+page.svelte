<script lang="ts">
    import * as nv from "nervous";
    import { browser } from "$app/environment";

    import tests from "./tests";
    import { afterUpdate, onMount } from "svelte";

    let testResults: any[] = [];

    let randomValues: any[] = [];

    const runTests = async () => {
        await nv.init();

        testResults = [];
        for (let i = 0; i < tests.length; i++) {
            let suite = tests[i];
            testResults.push({
                name: suite.suite,
                results: [],
            });

            for (let j = 0; j < suite.tests.length; j++) {
                let test = suite.tests[j];
                if (test === undefined) continue;
                testResults[i].results.push({
                    name: test.name,
                    res: null,
                });
                testResults = testResults;

                let codeErrored = false;
                const codeRes: any = await (async (): Promise<
                    any[] | string
                > => {
                    try {
                        return await test.code();
                    } catch (e) {
                        codeErrored = true;
                        return e as string;
                    }
                })();

                const codeResStrings =
                    codeRes.constructor === Array
                        ? codeRes.map((e) => JSON.stringify(e))
                        : "";

                let expectsErrored = false;
                const expectsRes = await (async (): Promise<any[] | string> => {
                    try {
                        return await test.expects();
                    } catch (e) {
                        expectsErrored = true;
                        return e as string;
                    }
                })();

                const expectsResStrings =
                    expectsRes.constructor === Array
                        ? expectsRes.map((e) => JSON.stringify(e))
                        : "";

                testResults[i].results[j] = {
                    ...testResults[i].results[j],
                    res: JSON.stringify(codeRes) == JSON.stringify(expectsRes),
                    code: test.code,
                    showCode: false,
                    expects: test.expects,
                    codeErrored,
                    expectsErrored,
                    codeRes,
                    expectsRes,
                    codeResStrings,
                    expectsResStrings,
                };
            }
        }
    };

    if (browser) {
        runTests();
        const main = async () => {
            await nv.init();

            // let times = [];
            // const a = await nv.random([2048, 2047]);
            // let s;
            // for (let i = 0; i < 5; i++) {
            //     const p1 = performance.now();
            //     s = await a.sum();
            //     const p2 = performance.now();
            //     // await s.print()
            //
            //     times.push(p2 - p1);
            // }
            // await s.print();
            // console.log(times.reduce((a, b) => a + b, 0) / times.length);
        };
        main();
    }
    let randomLoaded = false;

    const thingy = async (el) => {
        const c = document.getElementById("canvas") as HTMLCanvasElement;
        if (c === null) return;
        let ctx = c.getContext("2d", { willReadFrequently: true });
        if (ctx === null) return;
        c.width = c.getBoundingClientRect().width;
        c.height = c.getBoundingClientRect().height;

        let scaleFactor = 3;
        c.width = Math.ceil(c.width * scaleFactor);
        c.height = Math.ceil(c.height * scaleFactor);
        ctx.scale(scaleFactor, scaleFactor);

        let canvasWidth = c.width;
        let canvasHeight = c.height;

        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        let id = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
        let pixels = id.data;
        randomValues = await nv.random([canvasWidth, canvasHeight]).values();
        for (let i = 0; i < canvasWidth; i++) {
            for (let j = 0; j < canvasHeight; j++) {
                let off = (j * id.width + i) * 4;
                // pixels[off] = randomValues[i][j] * 255;
                pixels[off + 1] = randomValues[i][j] * 255;
                // pixels[off + 2] = randomValues[i][j][2] * 255;
                pixels[off + 3] = 255;
            }
        }
        ctx.putImageData(id, 0, 0);
    };
</script>

<head>
    <title>Tests</title>
</head>

<body class="max-w-3xl mx-auto mb-12 mt-4 text-stone-900">
    <button
        class="rounded shadow text-white bg-green-500 active:bg-green-600 p-2"
        on:click={() => {
            runTests;
        }}
    >
        Run Tests
    </button>

    {#if testResults.length !== 0}
        <button
            class="rounded shadow text-white bg-red-500 active:bg-red-600 p-2"
            on:click={() => {
                testResults = [];
            }}
        >
            Clear Results
        </button>
    {/if}

    {#each testResults as suite}
        {#if suite}
            <h3 class="text-xl dark:text-slate-200 font-medium mt-4">
                {suite.name}
            </h3>
            <div class="max-w-xl mx-auto">
                {#each suite.results || [] as result}
                    <div
                        class="my-4 ring ring-red-600 dark:ring-red-400 rounded flex flex-col"
                    >
                        <p class="py-1 dark:text-slate-200 pl-4 font-medium">
                            {result.name}
                        </p>
                        {#if result.name === "random()"}
                            <canvas id="canvas" use:thingy />
                        {/if}
                        <div class="flex flex-row">
                            {#if result.res === null}
                                <div class="flex ml-auto mb-2">
                                    <svg
                                        class="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-700"
                                        xmlns="http://www.w3.org/2000/svg"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                    >
                                        <circle
                                            class="opacity-30"
                                            cx="12"
                                            cy="12"
                                            r="10"
                                            stroke="currentColor"
                                            stroke-width="4"
                                        />
                                        <path
                                            class="opacity-100"
                                            fill="currentColor"
                                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                        />
                                    </svg>
                                    <p class="leading-none">running...</p>
                                </div>
                            {:else if result.codeErrored || result.expectsErrored}
                                {#if result.codeErrored}
                                    <div class="bg-red-300 flex-grow pl-1">
                                        ❌ Test {result.codeRes}
                                    </div>
                                {/if}
                                {#if result.expectsErrored}
                                    <div class="bg-red-300 flex-grow pl-1">
                                        ❌ Expected Result Calculation {result.expectsRes}
                                    </div>
                                {/if}
                            {:else}
                                <div class="flex flex-col w-full">
                                    {#each result.codeRes as _r, i}
                                        {#if result.codeResStrings[i] == result.expectsResStrings[i]}
                                            <div
                                                class="bg-green-300 {result.showCode
                                                    ? ''
                                                    : 'last:rounded-bl'}  pl-1"
                                            >
                                                ✅ {result.codeResStrings[i]} ≡ {result
                                                    .expectsResStrings[i]}
                                            </div>
                                        {:else}
                                            <div class="bg-red-300 w-full pl-1">
                                                ❌ {result.codeResStrings[i]} ≢ {result
                                                    .expectsResStrings[i]}
                                            </div>
                                        {/if}
                                    {/each}
                                </div>
                            {/if}
                            <button
                                class=" ml-auto mr-0 {result.showCode
                                    ? ''
                                    : 'rounded-br'} shadow text-stone-900 bg-stone-100 active:bg-stone-300  font-bold leading-none w-12 "
                                on:click={() =>
                                    (result.showCode = !result.showCode)}
                                >{result.showCode ? "↑" : "↓"}</button
                            >
                        </div>
                        <pre
                            class="transition-all ease-in-out px-4 {result.showCode
                                ? 'max-h-96 py-4'
                                : 'max-h-0'}  overflow-scroll rounded-b  text-sm bg-stone-900 text-stone-100">{result.code}</pre>
                    </div>
                {/each}
            </div>
        {/if}
    {/each}
</body>
