---
title: Introduction
description: Docs intro
layout: ../../layouts/MainLayout.astro
---

## nervous

**A minimal, functional, ML framework**

Take what you need ML framework for the browser. Providing example models and
the components to easily create your own.

Minimal and comprehensible functional ML library, with GPU acceleration using
WebGPU _<sup>WebGPU is currently available behind a flag on
[Chrome Canary](https://www.google.com/chrome/canary/) and
[Firefox Nightly (untested)](https://www.mozilla.org/en-US/firefox/channel/desktop/)
</sup>_.

## Getting Started

```bash
# hasn't been published yet
$ npm i @typio/nervous # or pnpm or yarn
```

## Basics

It is important to remember that when working with the webgpu backend nearly
every function is async. This is necessary because the webgpu API uses async.
