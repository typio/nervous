<script lang="ts">
    import "../app.css";

    import { writable } from "svelte/store";
    import { browser } from "$app/environment";
    import { onMount } from "svelte";

    const defaultValue = "dark";
    const initialValue = browser
        ? window.localStorage.getItem("theme") ?? defaultValue
        : defaultValue;
    export const theme = writable<string>(initialValue);
    theme.subscribe((value) => {
        if (browser) {
            window.localStorage.setItem("theme", value);
        }
    });

    $: if ($theme === "dark") {
        if (browser) document.documentElement.classList.add("dark");
    } else if (browser) document.documentElement.classList.remove("dark");

    onMount(() => {
        if (browser) document.getElementById("theme-toggle")?.classList.remove("scale-0");
    });
</script>

<head>
    <meta content="text/html;charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="shortcut icon" href="../stolen-favicon.png" type="image/png" />

    <title>nervous</title>;
</head>

<nav class="mt-8 w-full flex">
    <a class="ml-6 text-3xl text-red-600" href="/">nervous</a>

    <button
        aria-label="Toggle dark mode"
        id="theme-toggle"
        class="ml-auto mr-6 text-2xl w-12 scale-0 transition-[transform] ease-in-out duration-300"
        on:click={() => {
            $theme === "light" ? theme.set("dark") : theme.set("light");
        }}
    >
        {#if $theme === "light"}
            <img
                alt="sun"
                src="sun-with-face.png"
            />
        {:else}
            <img
                alt="moon"
                src="new-moon-face.png"
            />
        {/if}
    </button>
</nav>

<slot />
