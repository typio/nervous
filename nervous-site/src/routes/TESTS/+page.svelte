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
                let test = suite.tests[j];
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
            await nv.init({ backend: "webgpu" });

            let k = nv.tensor([[ 4.,  0.,  0.,  3.],
        [ 3.,  2.,  6., 10.],
        [ 8.,  6.,  4.,  9.],
        [ 2.,  4.,  5.,  1.],
        [ 1.,  7.,  0.,  2.],
        [ 7.,  5.,  7.,  8.]])

let j = nv.tensor([2., 7., 7., 3.])


let n = nv.tensor([[43.,  5., 32., 79., 97., 89., 21., 33., 70., 81.],
        [70.,  9., 16., 94., 64.,  9., 45., 73., 33., 87.],
        [67., 66., 57., 62., 23., 48., 29., 41., 86., 88.],
        [45., 35., 78., 63.,  3., 33., 24., 14., 26.,  7.],
        [60., 38., 73., 27., 70., 64.,  1., 39., 33., 84.],
        [63., 14., 17.,  0.,  7., 74., 93., 65., 57., 78.],
        [43., 85., 57., 14., 14., 61., 83., 45., 60.,  6.],
        [27., 38.,  1., 56., 97., 91., 12., 11., 42., 52.]])

            console.log(await (await n.mod(40)).flatValues(3))
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
                                                ❌ {result.codeRes[i]} ≢ {result
                                                    .expectsRes[i]}
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
