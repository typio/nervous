import { test } from 'uvu';
import * as assert from 'uvu/assert';

import * as t from '../src/tensor'

test('tensor3d', () => {
    let tensor = t.tensor3d([[[0, 1, 0]], [[0, 0, 4]]])
    assert.equal(tensor.rank, 3)
    assert.equal(tensor.shape, [2, 1, 3])
})

test.run();