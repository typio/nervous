import p5 from "p5"

import * as nv from "nervous"

const s = (p) => {
    let x_range = [-1, 10]
    let y_range = [-1, 10]

    let points: [{ x: number, y: number }] = []
    let m = 1

    p.setup = () => {
        p.createCanvas(512, 512)
        let button = p.createButton('Clear Points')
        button.parent('p5UI')
        button.mousePressed(clearPoints)
    }

    p.draw = () => {
        // clear and draw background
        p.clear(0)
        p.background(244, 244, 255)

        // draw graph ticks
        for (let i = x_range[0]; i < x_range[1]; i++) {
            p.line(i)

        }

        // draw points
        for (let i = 0; i < points.length; i++) {
            p.circle(points[i].x, points[i].y, 5)
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

    // p.windowResized = () => {
    //     p.resizeCanvas(p.windowWidth, p.windowHeight)
    // }
}

new p5(s, document.getElementById('p5CanvasParent') ?? undefined)

