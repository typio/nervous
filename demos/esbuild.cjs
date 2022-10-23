#!/usr/bin/env node
const fs = require('fs')
let files = []

const getFiles = (path) => {
  if (fs.lstatSync(path).isDirectory()) { // is this a folder?
    fs.readdirSync(path).forEach(f => {   // for everything in this folder
      getFiles(path + '/' + f)            // process it recursively
    })
  } else if (path.endsWith(".ts")) {  // is this a file we are searching for?
    files.push(path)                  // record it
  }
}

getFiles("src")

require("esbuild")
  .build({
    logLevel: "info",
    entryPoints: files,
    bundle: true,
    // minify: true,
    loader: { ".data": "binary" },
    outdir: "src/dist/",
    watch: true,
  })
  .catch(() => process.exit(1))