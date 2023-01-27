import { sveltekit } from "@sveltejs/kit/vite";
import fs from "fs";

/** @type {import('vite').Plugin} */
const binaryLoader = {
    name: "hex-loader",
    transform(code, id) {
        const [path, query] = id.split("?");
        if (query != "hex") return null;

        const data = fs.readFileSync(path);
        const hex = data.toString("hex");

        return `export default '${hex}';`;
    },
};

/** @type {import('vite').UserConfig} */
const config = {
    plugins: [binaryLoader, sveltekit()],
    test: {
        include: ["src/**/*.{test,spec}.{js,ts}"],
    },
    server: {
        port: 3333,
    },
};

export default config;
