import { Tensor } from '../tensor'

export const rank = async (_a: Tensor) => {
	let a: Tensor = _a.usingGPUBuffer ? await _a.toJS() : _a
	if (a.data.length === 2) return 0 // scalar
	let rank = 0
	for (let i = 0; i < 4; i++) {
		if (a.data[i] !== 0) rank++
		else return rank
	}
	return rank
}
