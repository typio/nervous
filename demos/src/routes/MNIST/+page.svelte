<script lang="ts">
	import P5 from 'p5-svelte';
	import type { Sketch } from 'p5-svelte/types';

	import * as lt from 'light';

	import pako from 'pako';
	import { browser } from '$app/environment';
    import { onMount } from 'svelte';


	import trainImagesFile from './t10k-images.idx3-ubyte?raw';
	import labelLabelsFile from './t10k-labels.idx1-ubyte?raw';

	let sketch: Sketch;
	onMount(async () => {
		let pixelValues = [];

		let dataFileBuffer = new ArrayBuffer(trainImagesFile.length);
		let labelFileBuffer = new ArrayBuffer(labelLabelsFile.length);

		for (let i = 0; i < trainImagesFile.length; i++) {
			dataFileBuffer[i] = trainImagesFile.charCodeAt(i);
			labelFileBuffer[i] = labelLabelsFile.charCodeAt(i);
		}

		for (let image = 0; image < 100; image++) {
			let pixels = [];

			for (let x = 0; x <= 27; x++) {
				for (let y = 0; y <= 27; y++) {
					pixels.push(dataFileBuffer[image * 28 * 28 + (x + y * 28) + 15]);
				}
			}

			let imageData = {};
			imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels;

			pixelValues.push(imageData);
		}

		sketch = (p5) => {
			let gp;
			p5.setup = () => {
				p5.createCanvas(512, 512);
				gp = p5.createGraphics(p5.width, p5.height);
				gp.pixelDensity(1);
				p5.noLoop();
			};

			p5.draw = () => {
				p5.clear(0, 0, 0, 0);
				for (let digit = 0; digit < 10; digit++) {
					let randomIndex = Math.floor(Math.random() * 100);

					gp.loadPixels();
					for (var x = 0; x < 28; x++) {
						for (var y = 0; y < 28; y++) {
							var index = (p5.width * y + x) * 4;
							gp.pixels[index + 0] = Object.values(pixelValues[randomIndex])[0][28 * y + x];
							gp.pixels[index + 1] = Object.values(pixelValues[randomIndex])[0][28 * y + x];
							gp.pixels[index + 2] = Object.values(pixelValues[randomIndex])[0][28 * y + x];
							gp.pixels[index + 3] = 255;
						}
					}
					gp.updatePixels();
					p5.image(gp, 0 + digit*28, 0, p5.width, p5.width);
					p5.text(Object.keys(pixelValues[randomIndex])[0], 10 + digit*28, 50);
				}
			};
		};
	})
</script>

<div class="sketch-1">
	<P5 {sketch} />
</div>

<style>
</style>
