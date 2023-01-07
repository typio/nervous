<script lang="ts">
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
                const codeResString = JSON.stringify(codeRes);

                let expectsErrored = false;
                const expectsRes = await (async () => {
                    try {
                        return await test.expects();
                    } catch (e) {
                        expectsErrored = true;
                        return e;
                    }
                })();
                const expectsResSting = JSON.stringify(expectsRes);

                testResults[i].results[j] = {
                    ...testResults[i].results[j],
                    res: codeResString === expectsResSting,
                    code: test.code,
                    showCode: false,
                    expects: test.expects,
                    codeErrored,
                    expectsErrored,
                    codeRes,
                    expectsRes,
                    codeResString,
                    expectsResSting,
                };
            }
        }
    };

    if (browser) {
        runTests();
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
            // let a = nv.tensor([1, 2]);
            // let b = nv.tensor([4, 4]);
            // let result = await a.add(b);

            const [n, m] = await Promise.all([
                nv.random([2047, 512]),
                nv.random([512, 2047]),
            ]);
            let res = await n.matmul(m);

            console.log("hi", await res.flatValues());

            // console.log(await result.values());

            // let start = performance.now();
            // const [n, m] = await Promise.all([
            //     nv.random([2047, 2047]),
            //     nv.random([2047, 2047]),
            // ]);
            // const n = await nv.random([2047, 2047])
            // const m = await nv.random([2047, 2047])
            // let end = performance.now();
            // console.log(
            //     `%c${end - start}ms`,
            //     "background: dodgerblue; color: white; padding: 3px 4px; border-radius:2px;"
            // );

            // let res = await n.minus(m);
            // console.log(res);

            // console.log(await res.values());
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

<body class="max-w-3xl mx-auto mb-12 mt-4 text-stone-900">
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

    <button
        class="rounded shadow text-white bg-green-500 active:bg-green-600 p-2"
        on:click={runTests}
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
            <h3 class="text-xl font-medium mt-4">{suite.name}</h3>
            <div class="max-w-xl mx-auto">
                {#each suite.results || [] as result}
                    <div class="my-4 ring ring-red-300 rounded flex flex-col">
                        <p class="py-1 pl-4 font-medium">{result.name}</p>
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
                            {:else}
                                <!-- <p>{result.res}</p> -->

                                {#if result.codeErrored || result.expectsErrored}
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
                                {:else if result.res}
                                    <div class="w-full flex flex-wrap flex-col">
                                        {#each result.codeRes as r, i}
                                            <div class="bg-green-300 {result.showCode
                                                ? ''
                                                : 'last:rounded-bl'}  pl-1">
                                                ✅ {result.codeRes[i]} ≡ {result
                                                    .expectsRes[i]}
                                            </div>
                                        {/each}
                                    </div>
                                {:else}
                                    <div class="bg-red-300 w-full pl-1">
                                        ❌ {result.codeRes} ≢ {result.expectsRes}
                                    </div>
                                {/if}
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
