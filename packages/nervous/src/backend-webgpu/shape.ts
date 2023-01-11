import { Tensor } from '../tensor'
import { toArr } from '../tensorUtils'

export const shape = async (_a: Tensor): Promise<number[]> => {
  // remove trailing 1's in shape segement of data
  let a: Tensor
  if (_a.usingGPUBuffer) a = await _a.toJS()
  else a = _a
  let i = 3
  while (i > 0 && a.data[i] === 0) i--

  return toArr(a.data.slice(0, i + 1))
}
