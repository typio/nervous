import p5 from "p5"

import * as nv from "nervous"

import fishes from './fish.json'

console.log(nv.random([2, 3]).getValues());


const s = (p) => {
    const fishColors = new Map([
        ['Bream', 'dodgerblue'],
        ['Pike', 'mediumvioletred'],
        ['Whitefish', 'gold'],
        ['Parkki', 'mediumaquamarine'],
        ['Perch', 'lime'],
        ['Smelt', 'blue'],
        ['Roach', 'red']
    ]);

    let x_range = [-300, 1800]
    let y_range = [-10, 70]

    let points: [{ x: number, y: number }] = []
    let m = 1


    p.setup = () => {
        p.createCanvas(Math.min(400, p.windowWidth * .8), Math.min(400, p.windowHeight * .7))
        p.textAlign(p.CENTER, p.CENTER);
    }

    p.draw = () => {
        // clear and draw background
        p.clear(0)
        p.background(244, 244, 255)

        p.stroke(0, 0, 0, 255)
        p.line(
            p.map(0, x_range[0], x_range[1], 0, p.width),
            p.height,
            p.map(0, x_range[0], x_range[1], 0, p.width),
            0
        )
        p.line(
            0,
            p.map(0, y_range[0], y_range[1], p.height, 0),
            p.width,
            p.map(0, y_range[0], y_range[1], p.height, 0)
        )

        // draw graph ticks
        p.stroke(0, 0, 0, 50)
        p.fill(0, 0, 0, 150)
        let xnum_yPos = p.map(0, y_range[0], y_range[1], p.height, 0) + 10
        for (let i = x_range[0]; i <= x_range[1]; i += 150) {
            p.line(
                p.map(i, x_range[0], x_range[1], 0, p.width),
                p.height,
                p.map(i, x_range[0], x_range[1], 0, p.width),
                0
            )
            p.textAlign(p.CENTER, p.TOP)
            p.text(i,
                p.map(i, x_range[0], x_range[1], 0, p.width),
                xnum_yPos)
        }
        p.text('Weight (g)', p.width / 2, p.height - 20)

        let ynum_xPos = p.map(0, x_range[0], x_range[1], 0, p.width) - 10
        for (let i = y_range[0]; i <= y_range[1]; i += 10) {
            p.line(
                0,
                p.map(i, y_range[0], y_range[1], p.height, 0),
                p.width,
                p.map(i, y_range[0], y_range[1], p.height, 0)
            )
            p.textAlign(p.RIGHT, p.CENTER)
            p.text(i,
                ynum_xPos,
                p.map(i, y_range[0], y_range[1], p.height, 0)
            )
        }
        let textGraphics = p.createGraphics(100, 100)
        textGraphics.rotate(Math.PI / 2)
        let t = textGraphics.text('Length (cm)', 0, 0)
        p.image(textGraphics, 10, p.height / 2 - t.width / 2, 100, 100)

        // draw points
        // for (let i = 0; i < points.length; i++) {
        //     p.fill(p.color('black'))
        //     p.circle(points[i].x, points[i].y, 5)
        // }

        for (let i = 0; i < fishes.length; i++) {
            p.stroke(0, 0, 0, 0)
            p.fill(p.color(fishColors.get(fishes[i].Species)))
            p.circle(
                p.map(fishes[i].Weight, x_range[0], x_range[1], 0, p.width),
                p.map(fishes[i].Length1, y_range[0], y_range[1], p.height, 0), 6)
        }

        // draw line
        p.stroke(255, 0, 0)
        p.line(
            p.map(x_range[0], x_range[0], x_range[1], 0, p.width),
            p.map(y_range[0] * m, y_range[0], y_range[1], p.height, 0),
            p.map(x_range[1], x_range[0], x_range[1], 0, p.width),
            p.map(y_range[1] * m, y_range[0], y_range[1], p.height, 0))

        // draw legend
        p.stroke(0, 0, 0, 0)
        p.fill(p.color('white'))
        p.rect(p.width - 100, 10, 90, 170)

        let i = 0
        fishColors.set('Custom', 'black').forEach((color, fish) => {
            p.fill(p.color(color))
            p.rect(p.width - 90, 20 + i * 20, 10, 10)
            p.textAlign(p.LEFT, p.CENTER)
            p.text(fish, p.width - 70, 25 + i * 20)
            i++
        });
    }

    p.mouseClicked = () => {
        fishes.push({
            Height: 0,
            Length1: Math.max(0, p.map(p.mouseY, p.height, 0, y_range[0], y_range[1])),
            Length2: 0,
            Length3: 0,
            Species: "Custom",
            Weight: Math.max(0, p.map(p.mouseX, 0, p.width, x_range[0], x_range[1])),
            Width: 0
        })
    }

    let clearPoints = () => {
        for (let i = 0; i < fishes.length; i++) {
            if (fishes[i].Species === 'Custom') {
                fishes.splice(i, 1);
                i--;
            }
        }
    }

    document.getElementById('clear-points-btn').onclick = clearPoints


    let toY = () => {

    }

    p.windowResized = () => {
        p.resizeCanvas(Math.min(512, p.windowWidth * .8), Math.min(512, p.windowHeight * .7))

    }
}

new p5(s, document.getElementById('p5CanvasParent') ?? undefined)

