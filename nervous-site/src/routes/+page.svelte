<script lang="ts">
    import { browser } from "$app/environment";
    import * as nv from "nervous";

    if (browser) {
        const main = async () => {
            await nv.init();

            const a = nv.tensor([
                [
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                ],
                [
                    [9, 10, 11],
                    [12, 13, 14],
                    [15, 16, 17],
                ],
                [
                    [18, 19, 20],
                    [21, 22, 23],
                    [24, 25, 26],
                ],
            ]);

            // a.slice([0, 0, 0], [1, 3, 2]).print();
            console.log(a.strides)
            console.log(a.shape)
            a.print()
        };
        main();
    }
</script>

<head>
    <title>Demos</title>
</head>

<div class="max-w-5xl mx-auto">
    <p class="px-4 pt-4 text-red-900">
        Demonstrations of the <a
            href="https://github.com/typio/nervous"
            class="font-medium text-red-500 hover:border-b-4 border-red-500"
            >nervous</a
        > machine learning framework.
    </p>
    <div class="flex flex-wrap">
        <div
            class="rounded shadow dark:shadow-none dark:border-red-500 dark:border-2  p-6 mt-4 mx-4 w-screen sm:w-auto max-w-sm"
        >
            <a
                href="/TESTS"
                class="font-medium text-red-500 hover:border-b-4 border-red-500"
                >Tests</a
            >
            <p class="text-red-900 text-sm">
                Library tests. Occasionally with some visualization and
                benchmarks.
            </p>
        </div>
        <div
            class="rounded shadow dark:shadow-none dark:border-red-500 dark:border-2  p-6 mt-4 mx-4 w-screen sm:w-auto max-w-sm"
        >
            <a
                href="/IRIS"
                class="font-medium text-red-500 hover:border-b-4 border-red-500"
                >Iris Classifier</a
            >
            <p class="text-red-900 text-sm">
                Linear classifier for the famous Iris dataset.
            </p>
        </div>
        <div
            class="rounded shadow dark:shadow-none dark:border-red-500 dark:border-2 p-6 mt-4 mx-4 w-screen sm:w-auto max-w-sm"
        >
            <a
                href="/IRIS2"
                class="font-medium text-red-500 hover:border-b-4 border-red-500"
                >Iris Classifier v2</a
            >
            <p class="text-red-900 text-sm">
                Linear classifier for the famous Iris dataset, redone with a
                implementation focusing on WebGPU.
            </p>
        </div>
        <div
            class="rounded shadow dark:shadow-none dark:border-red-500 dark:border-2 p-6 mt-4 mx-4 w-screen sm:w-auto max-w-sm"
        >
            <a
                href="/MNIST"
                class="font-medium text-red-500 hover:border-b-4 border-red-500"
                >MNIST digit recognition</a
            >
            <p class="text-red-900 text-sm">Working on this...</p>
        </div>
        <div
            class="rounded shadow dark:shadow-none dark:border-red-500 dark:border-2  p-6 mt-4 mx-4 w-screen sm:w-auto max-w-sm"
        >
            <a
                href="/LINREG"
                class="font-medium text-red-500 hover:border-b-4 border-red-500"
                >Linear Regression</a
            >
            <p class="text-red-900 text-sm">Working on this...</p>
        </div>
        <div
            class=" rounded shadow dark:shadow-none dark:border-red-500 dark:border-2  p-6 mt-4 mx-4 w-screen sm:w-auto max-w-sm"
        >
            <a
                href="/XOR"
                class=" font-medium text-red-500 hover:border-b-4 border-red-500"
                >XOR</a
            >
            <p class="text-red-900 text-sm">Planned for future...</p>
        </div>
    </div>
</div>
