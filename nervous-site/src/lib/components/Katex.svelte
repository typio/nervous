<script>
    import katex from "katex";
    export let options = {
        throwOnError: false,
    };
    export let math = "a^2";
</script>

<head>
    <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css"
        integrity="sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X"
        crossorigin="anonymous"
    />
</head>

{@html katex.renderToString(math, options)}
