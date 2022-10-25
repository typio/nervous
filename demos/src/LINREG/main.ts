import p5 from "p5"

import * as nv from "nervous"

import fish from './fish.json'

const s = (p) => {
    let x_range = [-150, 1800]
    let y_range = [-10, 70]

    let points: [{ x: number, y: number }] = []
    let m = 1

    p.setup = () => {
        p.createCanvas(Math.min(512, p.windowWidth * .8), Math.min(512, p.windowHeight * .7))
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
        for (let i = x_range[0]; i <= x_range[1]; i += 150) {
            p.line(
                p.map(i, x_range[0], x_range[1], 0, p.width),
                p.height,
                p.map(i, x_range[0], x_range[1], 0, p.width),
                0
            )
            p.text(i,
                p.map(i, x_range[0], x_range[1], 0, p.width),
                p.height - 25)
        }

        for (let i = y_range[0]; i <= y_range[1]; i += 10) {
            p.line(
                0,
                p.map(i, y_range[0], y_range[1], p.height, 0),
                p.width,
                p.map(i, y_range[0], y_range[1], p.height, 0)
            )
            p.text(i,
                25,
                p.map(i, y_range[0], y_range[1], p.height, 0)
            )

        }

        // draw points
        for (let i = 0; i < points.length; i++) {
            p.circle(points[i].x, points[i].y, 5)
        }

        for (let i = 0; i < fish.length; i++) {
            p.stroke(0, 0, 0, 0)
            if (fish[i].Species === 'Bream') {
                p.fill(p.color('dodgerblue'))
            } else if (fish[i].Species === 'Pike') {
                p.fill(p.color('magenta'))
            } else if (fish[i].Species === 'Whitefish') {
                p.fill(p.color('sandybrown'))
            } else if (fish[i].Species === 'Parkki') {
                p.fill(p.color('darkgray'))
            } else if (fish[i].Species === 'Perch') {
                p.fill(p.color('limegreen'))
            } else if (fish[i].Species === 'Smelt') {
                p.fill(p.color('powderblue'))
            }
            p.circle(
                p.map(fish[i].Weight, x_range[0], x_range[1], 0, p.width),
                p.map(fish[i].Length1, y_range[0], y_range[1], p.height, 0), 6)
        }

        // draw line
        p.stroke(255, 0, 0)
        p.line(
            p.map(x_range[0], x_range[0], x_range[1], 0, p.width),
            p.map(y_range[0] * m, y_range[0], y_range[1], p.height, 0),
            p.map(x_range[1], x_range[0], x_range[1], 0, p.width),
            p.map(y_range[1] * m, y_range[0], y_range[1], p.height, 0))
    }

    p.mouseClicked = () => {
        points.push({ x: p.mouseX, y: p.mouseY })
    }

    let clearPoints = () => {
        points = []
    }

    document.getElementById('clear-points-btn').onclick = clearPoints


    let toY = () => {

    }

    p.windowResized = () => {
        p.resizeCanvas(Math.min(512, p.windowWidth * .8), Math.min(512, p.windowHeight * .7))

    }
}

new p5(s, document.getElementById('p5CanvasParent') ?? undefined)

