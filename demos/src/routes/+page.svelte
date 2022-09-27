<script lang="ts">
	import P5 from 'p5-svelte';
	import type { Sketch } from 'p5-svelte/types';

	import * as light from 'light'

	light.tensor([1,2,3]).print()

	let range1Val = 0.5;
	let range2Val = 255;

	const sketch: Sketch = (p5) => {
		p5.setup = () => {
			p5.createCanvas(512, 512);
		};

		p5.draw = () => {
			p5.clear(0, 0, 0, 0);
			p5.background(233);
			p5.noStroke();
			p5.fill(255, 80, 0, range2Val);
			p5.ellipse(p5.mouseX, p5.mouseY, 150 * range1Val, 150 * range1Val);
		};
	};
</script>

<h1>No machine learning here yet...</h1>

<div class="sketch-1">
	<P5 {sketch} />
	<div class="controls-1">
		<label for="range-1">Circle Radius</label>
		<input type="range" name="" id="range-1" step="0.01" min="0" max="1" bind:value={range1Val} />
		<label for="range-2">Circle Opacity</label>
		<input type="range" name="" id="range-2" min="0" max="255" bind:value={range2Val} />
	</div>
</div>

<style>
	.sketch-1 {
		border: 2px solid;
		width: fit-content;
	}

	.controls-1 {
		display: flex;
		place-content: center;
		background-color: aqua;
	}
</style>
