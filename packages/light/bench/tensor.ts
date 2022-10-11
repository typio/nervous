import Benchmark from 'benchmark'

import * as lt from '../src/index'

let suite = new Benchmark.Suite

/* [100, 1000]
light:bench: dot x 9.00 ops/sec ±4.16% (26 runs sampled)
light:bench: dot2 x 10.14 ops/sec ±2.05% (30 runs sampled)
light:bench: Fastest is dot2
*/

/*
light:bench: nestedArr x 812 ops/sec ±0.29% (93 runs sampled)
light:bench: flatArr x 832 ops/sec ±0.16% (93 runs sampled)
light:bench: Fastest is flatArr
*/

/* [1000, 10000]
light:bench: floatArr x 86.36 ops/sec ±0.12% (73 runs sampled)
light:bench: flatArr x 85.09 ops/sec ±1.84% (73 runs sampled)
light:bench: Fastest is floatArr
 */

/* len=10, Math.random() > 0.5 ? [1, 2, 3] : 123
light:bench: isArray x 31,073,335 ops/sec ±0.12% (96 runs sampled)
light:bench: instanceOf x 41,566,630 ops/sec ±1.41% (93 runs sampled)
light:bench: constructor x 50,634,113 ops/sec ±0.33% (91 runs sampled)
light:bench: prototype.toString.call x 10,257,984 ops/sec ±0.31% (93 runs sampled)
light:bench: Fastest is constructor
*/

// TODO: Test new Array() and Array.fill() vs for 
let mixedTypes: (number[] | number)[] = new Array(10000).fill(0)
mixedTypes = mixedTypes.map(_ => Math.random() > 0.5 ? [1, 2, 3] : 123) 
console.log(mixedTypes) 

suite
    .add('isArray', () => {
        let arrayCount = 0
        for (let i = 0; i < mixedTypes.length; i++)
            if (Array.isArray(mixedTypes[i])) arrayCount++
        return arrayCount
    })
    .add('instanceOf', () => {
        let arrayCount = 0
        for (let i = 0; i < mixedTypes.length; i++)
            if (mixedTypes[i] instanceof Array) arrayCount++
        return arrayCount

    })
    .add('constructor', () => {
        let arrayCount = 0
        for (let i = 0; i < mixedTypes.length; i++)
            if (mixedTypes[i].constructor === Array) arrayCount++
        return arrayCount

    })
    .add('prototype.toString.call', () => {
        let arrayCount = 0
        for (let i = 0; i < mixedTypes.length; i++)
            if (Object.prototype.toString.call(mixedTypes[i]) === '[object Array]') arrayCount++
        return arrayCount

    })
    // add listeners
    .on('cycle', (event: { target: any }) => {
        console.log(String(event.target))
    })
    .on('complete', function () {
        console.log('Fastest is ' + this.filter('fastest').map('name'))
    })
    // run async
    .run({ 'async': true })

